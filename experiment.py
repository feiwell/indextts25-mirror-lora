import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
REPO = ROOT / 'repo'
sys.path.insert(0, str(REPO))
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ['HF_HOME'] = str(ROOT / 'hf-home')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
DATASET = 'mirror-verified-asr'
TESTS = ['今天下午三点，我们在地铁站旁边的咖啡店见面。',
         '冰箱里还有牛奶和面包，记得吃完早餐再出门。',
         '刚才那份文件已经发给你了，方便的时候看一下。',
         '外面开始下雨了，我把窗户关上，你先休息一会儿。',
         '明天的会议改到上午十点，地点还是二楼的小会议室。',
         '这个周末我想去公园散步，然后顺路买一点水果。']


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def seed(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def engine():
    from indextts.infer_v2_5 import IndexTTS2
    return IndexTTS2(model_dir=str(REPO / 'checkpoints'),
                     cfg_path=str(REPO / 'checkpoints/config.yaml'), device='cuda:0',
                     use_bf16=True, use_cuda_kernel=False, use_qwen_emo=False,
                     use_deepspeed=False, use_accel=False, use_torch_compile=False)


def reference():
    dest = ROOT / 'reference.wav'
    if not dest.exists():
        from webui_app.training import dataset as DS
        rows = json.loads((ROOT / 'curation-manifest-final.json').read_text())
        eligible = [r for r in rows if r.get('split') == 'train' and 5 <= r['duration'] <= 10]
        if not eligible:
            eligible = [r for r in rows if r.get('split') == 'train']
        best = max(eligible, key=lambda r: r['voiced_ratio'] - r.get('music_score', 0))
        shutil.copyfile(Path(DS.dir_of(DATASET)) / 'audio' / (best['id'] + '.wav'), dest)
        save('reference.json', best)
    return str(dest)


def synthesize(tts, label, limit=6):
    out = ROOT / 'listening' / label
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, text in enumerate(TESTS[:limit]):
        seed(20261010 + i)
        path = out / f'{i+1:02d}.wav'
        started = time.monotonic()
        tts.infer(reference(), text, str(path), lang='ZH', use_emo_text=False,
                  use_random=False, verbose=False, do_sample=True,
                  top_p=.8, top_k=30, temperature=.8, num_beams=3,
                  max_mel_tokens=600)
        import soundfile as sf
        audio, sr = sf.read(path)
        import numpy as np
        if len(audio) < sr or not np.isfinite(audio).all() or np.max(np.abs(audio)) < .001:
            raise RuntimeError('Invalid synthesized audio: ' + str(path))
        results.append({'path': str(path), 'text': text, 'seed': 20261010+i,
                        'seconds': time.monotonic()-started, 'duration': len(audio)/sr})
        print('SYNTHESIZED', label, i+1, results[-1], flush=True)
    save(f'listening-{label}.json', results)
    return results


def baseline_features():
    import torch
    from webui_app.training import dataset as DS, features as FT
    save('test-sentences.json', {'sentences': TESTS, 'postprocessing': 'none / raw'})
    from webui_app.training.reward import normalize_text
    training_texts = [normalize_text(u.text) for u in DS.load_meta(DATASET)]
    if any(normalize_text(text) in training_texts for text in TESTS):
        raise RuntimeError('Test sentence overlaps dataset')
    tts = engine()
    synthesize(tts, 'baseline')
    ex = FT.FeatureExtractor(tts=tts, device='cuda:0')
    dataset_dir = DS.dir_of(DATASET)
    items = DS.load_meta(DATASET)
    for i, u in enumerate(items):
        result = ex.extract_to(dataset_dir, u.id, u.audio_abs(dataset_dir), u.text, 'ZH')
        if not result.ok:
            raise RuntimeError(result.error)
        feat = torch.load(result.path, map_location='cpu', weights_only=False)
        u.has_features = True
        u.features_at = time.time()
        u.status = 'ready'
        u.n_codes = feat['n_codes']
        u.n_text_tokens = feat['n_text_tokens']
        u.mel_len = feat['mel_len']
        u.lang_token = feat['lang_token']
        if (i+1) % 10 == 0 or i+1 == len(items):
            DS.save_meta(DATASET, items)
            print('FEATURES', i+1, '/', len(items), flush=True)
    save('features-summary.json', FT.verify_dataset(DATASET))
    print('BASELINE AND FEATURES COMPLETE', flush=True)


def trainer(arch, run_name, probe=False):
    from webui_app.training import guard as GD, gpt_lora as GPT, cfm_lora as CFM
    cfg = GD.LoRAConfig(rank=8, alpha=16, dropout=.05, lr=5e-5,
                        weight_decay=.02, epochs=6, max_steps=300,
                        batch_size=2 if arch == 'gpt' else 1,
                        grad_accum=2, replay_ratio=0, replay_source='none',
                        eval_every=25, val_patience=4, keep_checkpoints=4,
                        seed=20261010, bf16=arch == 'gpt',
                        grad_checkpointing=arch == 'gpt')
    if probe:
        cfg.batch_size = 1
        cfg.grad_accum = 1
        cfg.max_steps = 4
    cfg.apply_target_preset('attn', arch)
    if arch == 'gpt':
        return GPT.GptTrainer(DATASET, cfg, GPT.GptTrainOptions(max_codes=320, max_text_tokens=200, log_every=5),
                              run_name=run_name, device='cuda:0', model_dir=str(REPO/'checkpoints'))
    return CFM.CfmTrainer(DATASET, cfg, CFM.CfmTrainOptions(max_prompt_frames=350,
                           max_target_frames=1050, max_total_frames=1400,
                           truncate_target=False, val_repeats=2, log_every=5),
                           run_name=run_name, device='cuda:0', model_dir=str(REPO/'checkpoints'))


def probes():
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from webui_app.training import gpt_lora as GPT, cfm_lora as CFM
    report = {}
    for arch in ['gpt', 'cfm']:
        torch.cuda.reset_peak_memory_stats()
        tr = trainer(arch, f'{arch}-gradient-probe', probe=True)
        tr.prepare(progress=lambda f, msg: print(arch, msg, flush=True))
        batches, _ = tr._epoch_batches(0)
        stats = []
        for batch in batches[:4]:
            torch.cuda.synchronize()
            started = time.monotonic()
            result = tr._train_micro_batch(batch)
            torch.cuda.synchronize()
            gradients = {n: float(p.grad.abs().max()) for n, p in tr.model.named_parameters()
                         if p.requires_grad and p.grad is not None}
            frozen_gradients = [n for n, p in tr.model.named_parameters() if not p.requires_grad and p.grad is not None]
            if not result or not gradients or max(gradients.values()) <= 0 or frozen_gradients:
                raise RuntimeError('Gradient probe failed for ' + arch)
            norm = tr._optimizer_step()
            stats.append({'loss': result['loss'], 'grad_norm': norm,
                          'max_abs_grad': max(gradients.values()),
                          'nonzero_gradient_tensors': sum(v > 0 for v in gradients.values()),
                          'seconds': time.monotonic()-started})
            print('PROBE', arch, stats[-1], flush=True)
        dest = ROOT / 'probe-adapters' / arch
        tr.model.save_pretrained(dest)
        state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(tr.model).items()}
        nonzero_B = {k: float(v.abs().max()) for k, v in state.items() if 'lora_B' in k}
        loader = GPT.load_base_gpt if arch == 'gpt' else CFM.load_base_cfm
        base, load_info = loader('cpu', str(REPO/'checkpoints'))
        loaded = PeftModel.from_pretrained(base, dest)
        reloaded = get_peft_model_state_dict(loaded)
        matches = set(state) == set(reloaded) and all(torch.equal(state[k], reloaded[k].cpu()) for k in state)
        if not matches or not nonzero_B or max(nonzero_B.values()) <= 0:
            raise RuntimeError('Adapter round trip failed for ' + arch)
        report[arch] = {'steps': stats, 'peak_vram_gb': torch.cuda.max_memory_allocated()/1e9,
                        'adapter_roundtrip_exact': matches, 'base_load': load_info,
                        'nonzero_B_tensors': sum(v > 0 for v in nonzero_B.values()),
                        'mean_step_seconds': sum(r['seconds'] for r in stats[1:])/max(1,len(stats)-1),
                        'notes': 'Probe weights are discarded; official training starts from base.'}
        save('gradient-probes.json', report)
        tr.release()
        del tr, base, loaded, state, reloaded
        gc.collect()
        torch.cuda.empty_cache()
    print('GRADIENT PROBES COMPLETE', flush=True)


def train():
    for arch in ['gpt', 'cfm']:
        tr = trainer(arch, f'{arch}-first-round')
        save(f'{arch}-training-config.json', {'config': tr.cfg.to_dict(), 'options': tr.options_dict()})
        result = tr.run(progress=lambda f, msg: print(arch, f'{f:.3f}', msg, flush=True))
        save(f'{arch}-training-report.json', asdict(result))
        if not result.ok:
            raise RuntimeError(result.error or 'Training not successful')
        print('TRAINING COMPLETE', arch, result.steps, result.seconds, flush=True)


def evaluate(labels=None):
    import torch
    from peft import PeftModel
    from webui_app.training import runs as RN
    best = {arch: RN.adapter_dir(f'{arch}-first-round') for arch in ['gpt', 'cfm']}
    alternate = {}
    for arch in best:
        candidates = sorted(RN.vault(f'{arch}-first-round', keep=4).list(), key=lambda c: c.metric)
        best_hash = hashlib.sha256((Path(best[arch]) / 'adapter_model.safetensors').read_bytes()).hexdigest()
        distinct = [c for c in candidates if hashlib.sha256(
            (Path(c.path) / 'adapter_model.safetensors').read_bytes()).hexdigest() != best_hash]
        if not distinct:
            raise RuntimeError('Need a checkpoint with different weights for ' + arch)
        alternate[arch] = distinct[0].path
    variants = {'finetuned-best': best, 'gpt-only': {'gpt': best['gpt']},
                'cfm-only': {'cfm': best['cfm']}, 'alternate-checkpoints': alternate}
    save('evaluation-adapters.json', variants)
    for label, adapters in variants.items():
        if labels is not None and label not in labels:
            continue
        tts = engine()
        if 'gpt' in adapters:
            tts.gpt = PeftModel.from_pretrained(tts.gpt, adapters['gpt']).eval()
        if 'cfm' in adapters:
            tts.s2mel.models['cfm'] = PeftModel.from_pretrained(tts.s2mel.models['cfm'], adapters['cfm']).eval()
        synthesize(tts, label)
        if label == 'finetuned-best':
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(tts.gpt.disable_adapter())
                stack.enter_context(tts.s2mel.models['cfm'].disable_adapter())
                synthesize(tts, 'adapter-disabled-control', limit=1)
            import numpy as np
            import soundfile as sf
            base_audio, base_sr = sf.read(ROOT / 'listening/baseline/01.wav')
            control, control_sr = sf.read(ROOT / 'listening/adapter-disabled-control/01.wav')
            tuned, tuned_sr = sf.read(ROOT / 'listening/finetuned-best/01.wav')
            same_shape = base_audio.shape == control.shape and base_sr == control_sr
            exact = same_shape and np.array_equal(base_audio, control)
            changed = tuned_sr != base_sr or tuned.shape != base_audio.shape or not np.array_equal(tuned, base_audio)
            save('adapter-inference-control.json', {'disabled_matches_baseline_exact': bool(exact),
                   'disabled_matches_shape': bool(same_shape),
                   'disabled_max_abs_difference': float(np.max(np.abs(base_audio-control))) if same_shape else None,
                   'enabled_changes_output': bool(changed),
                   'seed': 20261010})
            if not changed:
                raise RuntimeError('Enabled adapters did not change synthesized output')
        del tts
        gc.collect()
        torch.cuda.empty_cache()
    print('EVALUATION SYNTHESIS COMPLETE', flush=True)


def score():
    import numpy as np
    import soundfile as sf
    import whisper
    from rapidfuzz.distance import Levenshtein
    from webui_app.training.reward import RewardOptions, RewardScorer, normalize_text
    scorer = RewardScorer(RewardOptions(whisper_size='turbo', beam_size=5),
                          model_dir=str(REPO / 'checkpoints'))
    scorer._asr = whisper.load_model('turbo', device='cuda', download_root=str(ROOT / 'asr-models'))
    reference_embedding = scorer.embed(reference())
    manifest = json.loads((ROOT / 'curation-manifest-final.json').read_text())
    from webui_app.training import dataset as DS
    ds_dir = Path(DS.dir_of(DATASET))
    heldout = [r for r in manifest if r.get('split') == 'val']
    heldout_embeddings = [scorer.embed(str(ds_dir / 'audio' / (r['id'] + '.wav'))) for r in heldout]
    rows = []
    summaries = {}
    for label in ['baseline', 'finetuned-best', 'gpt-only', 'cfm-only', 'alternate-checkpoints']:
        synths = json.loads((ROOT / f'listening-{label}.json').read_text())
        for row in synths:
            hyp = scorer.transcribe(row['path'])
            target, actual = normalize_text(row['text']), normalize_text(hyp)
            edits = Levenshtein.distance(target, actual)
            embedding = scorer.embed(row['path'])
            audio, sr = sf.read(row['path'])
            row.update(label=label, asr_text=hyp, character_edits=edits, target_characters=len(target),
                       cer=edits/max(1, len(target)),
                       reference_similarity=scorer.cosine(reference_embedding, embedding),
                       heldout_similarity=float(np.mean([scorer.cosine(e, embedding) for e in heldout_embeddings])),
                       peak=float(np.abs(audio).max()), clipping_ratio=float((np.abs(audio) >= .999).mean()),
                       rms_dbfs=float(20*np.log10(np.sqrt((audio**2).mean())+1e-12)),
                       finite=bool(np.isfinite(audio).all()))
            rows.append(row)
            print('SCORE', label, row['cer'], row['reference_similarity'], flush=True)
        summaries[label] = {'corpus_cer': sum(r['character_edits'] for r in synths)/sum(r['target_characters'] for r in synths),
                            'mean_reference_similarity': float(np.mean([r['reference_similarity'] for r in synths])),
                            'mean_heldout_similarity': float(np.mean([r['heldout_similarity'] for r in synths])),
                            'samples': len(synths)}
        save('evaluation-metrics.json', {'summaries': summaries, 'rows': rows,
              'asr': 'Whisper turbo, temperature 0, beam 5, generic Chinese prompt',
              'speaker': 'CAMPPlus cosine to training reference and mean held-out clips',
              'limitations': 'Small private experiment; automated ASR/acoustic review only. These metrics do not establish perceptual naturalness or creator consent.'})
    scorer.unload()
    print('METRIC EVALUATION COMPLETE', summaries, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['baseline-features', 'probes', 'train', 'evaluate', 'evaluate-alternate', 'score'])
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    {'baseline-features': baseline_features, 'probes': probes, 'train': train, 'evaluate': evaluate,
     'evaluate-alternate': lambda: evaluate(['alternate-checkpoints']), 'score': score}[args.stage]()
