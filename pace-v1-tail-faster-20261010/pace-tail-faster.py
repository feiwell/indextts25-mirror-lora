import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tarfile

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
OUT = ROOT / 'pace-v1-tail-faster-20261010'
VERSION = ROOT / 'versions/mirror-joint-v1-20261010'
SOURCE = ROOT / 'pace-v1-20261010'


def verify_version():
    manifest = json.loads((VERSION / 'version.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((VERSION / name).read_bytes()).hexdigest() == expected, name
    return manifest


def main():
    manifest = verify_version()
    if OUT.exists():
        raise RuntimeError('Output already exists; choose a new directory.')
    OUT.mkdir()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    sys.path.insert(0, str(ROOT / 'repo'))
    import numpy as np
    import soundfile as sf
    import torch
    from peft import PeftModel
    from indextts.infer_v2_5 import IndexTTS2
    torch.set_num_threads(8)
    data = json.loads((SOURCE / 'metrics.json').read_text())
    hashes = json.loads((SOURCE / 'checksums.json').read_text())
    for name in ['pace-1.10.wav', 'pace-1.00.wav']:
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == hashes[name]
    slow, sr = sf.read(SOURCE / 'pace-1.10.wav', dtype='int16')
    default, default_sr = sf.read(SOURCE / 'pace-1.00.wav', dtype='int16')
    assert sr == default_sr == 22050
    assert len(slow) == 534528 + 4410 + 187904
    assert len(default) == 485888 + 4410 + 170752
    assert not slow[534528:538938].any() and not default[485888:490298].any()
    prefix = slow[:538938]
    tts = IndexTTS2(model_dir=str(ROOT / 'repo/checkpoints'),
                   cfg_path=str(ROOT / 'repo/checkpoints/config.yaml'),
                   device='cuda:0', use_bf16=True, use_cuda_kernel=False,
                   use_qwen_emo=False, use_deepspeed=False, use_accel=False,
                   use_torch_compile=False)
    tts.gpt = PeftModel.from_pretrained(tts.gpt, VERSION / 'adapters/gpt').eval()
    tts.s2mel.models['cfm'] = PeftModel.from_pretrained(
        tts.s2mel.models['cfm'], VERSION / 'adapters/cfm').eval()
    splitter = tts.split_text_by_tokens
    texts = []
    def tracked_split(text, max_tokens, lang_prefix=''):
        texts[:] = splitter(text, max_tokens, lang_prefix)
        return texts
    tts.split_text_by_tokens = tracked_split
    rows = []
    for factor in [1.0, .85, .75]:
        random.seed(20261022)
        np.random.seed(20261022)
        torch.manual_seed(20261022)
        torch.cuda.manual_seed_all(20261022)
        chunks = []
        native_path = OUT / f'native-{factor:.2f}.wav'
        generator = tts.infer(str(VERSION / 'reference.wav'), data['text'], str(native_path),
            lang='ZH', use_emo_text=False, use_random=False, verbose=False,
            stream_return=True, interval_silence=200, duration_factor=factor,
            max_text_tokens_per_segment=120, max_mel_tokens=1500,
            do_sample=True, top_p=.8, top_k=30, temperature=.8, num_beams=3)
        for emitted, chunk in enumerate(generator):
            if emitted % 2 == 0:
                chunks.append(chunk.to(torch.int16).numpy().reshape(-1))
        assert len(chunks) == len(texts) == 2
        native, native_sr = sf.read(native_path, dtype='int16')
        assert native_sr == sr
        assert np.array_equal(native, np.concatenate([chunks[0], np.zeros(4410, dtype=np.int16), chunks[1]]))
        if factor == 1.0:
            assert np.array_equal(chunks[1], default[490298:]), 'Default tail failed exact reproduction'
        tail_path = OUT / f'tail-{factor:.2f}.wav'
        full_path = OUT / f'adjusted-{factor:.2f}.wav'
        sf.write(tail_path, chunks[1], sr, subtype='PCM_16')
        combined = np.concatenate([prefix, chunks[1]])
        sf.write(full_path, combined, sr, subtype='PCM_16')
        reread, _ = sf.read(full_path, dtype='int16')
        assert np.array_equal(reread[:len(prefix)], prefix)
        assert np.array_equal(reread[len(prefix):], chunks[1])
        float_tail = chunks[1].astype(np.float64) / 32768
        assert np.isfinite(float_tail).all() and np.max(np.abs(float_tail)) > .001
        rows.append({'duration_factor': factor, 'full_path': full_path.name, 'tail_path': tail_path.name,
                     'tail_text': texts[1], 'tail_seconds': len(chunks[1]) / sr,
                     'full_seconds': len(combined) / sr,
                     'prefix_pcm_identical': True, 'native_tail_pcm_identical': True,
                     'clipping_ratio': float((np.abs(float_tail) >= .999).mean()), 'finite': True})
        print('FASTER TAIL GENERATED', rows[-1], flush=True)
    del tts
    gc.collect()
    torch.cuda.empty_cache()
    import whisper
    from rapidfuzz.distance import Levenshtein
    from webui_app.training.reward import normalize_text
    asr = whisper.load_model('turbo', device='cuda', download_root=str(ROOT / 'asr-models'))
    for row in rows:
        result = asr.transcribe(str(OUT / row['tail_path']), language='zh', temperature=0,
            beam_size=5, condition_on_previous_text=False, word_timestamps=True, fp16=True)
        expected = normalize_text(row['tail_text'])
        row.update(asr_text=result['text'].strip(),
            cer=Levenshtein.distance(expected, normalize_text(result['text'])) / len(expected),
            asr_segments=result['segments'])
        print('FASTER TAIL SCORED', row['duration_factor'], row['asr_text'], row['cer'], flush=True)
    verify_version()
    metadata = {'version': manifest['version'], 'text': data['text'], 'rows': rows,
        'tail_start_seconds': len(prefix) / sr, 'seed': 20261022, 'sample_rate': sr,
        'segment_budget': 120, 'max_mel_tokens': 1500,
        'default_tail_reproduced_exactly': True, 'preserved_version_verified_before_and_after': True,
        'source_hashes': {name: hashes[name] for name in ['pace-1.10.wav', 'pace-1.00.wav']},
        'method': 'Native duration-factor synthesis. Original factor-1.1 prefix plus 200ms silence retained exactly; final native segment replaced. No time stretching.',
        'limitations': 'Single excerpt and seed. ASR is an automated proxy, not human listening.'}
    (OUT / 'metrics.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
    (OUT / 'checksums.json').write_text(json.dumps(hashes, indent=2))
    archive_path = ROOT / 'pace-v1-tail-faster-results.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(OUT, arcname=OUT.name)
    print('FASTER TAIL COMPLETE', archive_path.stat().st_size,
          hashlib.sha256(archive_path.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
