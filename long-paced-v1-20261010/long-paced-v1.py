import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tarfile
import time

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
REPO = ROOT / 'repo'
OUT = ROOT / 'long-paced-v1-20261010'
VERSION = ROOT / 'versions/mirror-joint-v1-20261010'
SELECTED = ROOT / 'pace-v1-tail-selected-20261010'


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


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
    (OUT / 'segments').mkdir()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    sys.path.insert(0, str(REPO))
    import numpy as np
    import soundfile as sf
    import torch
    from peft import PeftModel
    from indextts.infer_v2_5 import IndexTTS2
    from webui_app.training.reward import normalize_text
    torch.set_num_threads(8)
    story = next(row for row in json.loads((ROOT / 'robustness-v1-20261010/texts.json').read_text())
                 if row['id'] == '03')
    paragraphs = story['text'].splitlines()
    old_rows = json.loads((ROOT / 'robustness-v1-20261010/metrics.json').read_text())['rows']
    baseline = next(row for row in old_rows if row['id'] == '03' and row['label'] == 'finetuned')
    selected = json.loads((SELECTED / 'metrics.json').read_text())
    selected_hashes = json.loads((SELECTED / 'checksums.json').read_text())
    first_path = SELECTED / 'adjusted-0.948.wav'
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == selected_hashes[first_path.name]
    first, sr = sf.read(first_path, dtype='int16')
    assert sr == 22050 and len(first) == 534528 + 4410 + 161792
    assert not first[534528:538938].any()
    first_texts = [baseline['segments'][0]['text'],
                  next(row for row in selected['rows'] if row['duration_factor'] == .948)['tail_text']]
    assert normalize_text(''.join(first_texts)) == normalize_text(paragraphs[0])
    rows, pieces = [], []
    silence = np.zeros(4410, dtype=np.int16)

    def add_segment(pcm, paragraph_index, segment_index, text, factor, token_count, source):
        path = f'segments/p{paragraph_index+1:02d}-s{segment_index+1:02d}.wav'
        sf.write(OUT / path, pcm, sr, subtype='PCM_16')
        float_pcm = pcm.astype(np.float64) / 32768
        assert np.isfinite(float_pcm).all() and np.max(np.abs(float_pcm)) > .001
        if pieces:
            pieces.append(silence)
        start = sum(len(piece) for piece in pieces) / sr
        pieces.append(pcm)
        rows.append({'paragraph': paragraph_index+1, 'segment': segment_index+1,
            'text': text, 'path': path, 'duration_factor': factor, 'text_tokens': token_count,
            'start_seconds': start, 'duration_seconds': len(pcm) / sr,
            'finite': True, 'clipping_ratio': float((np.abs(float_pcm) >= .999).mean()),
            'source': source})
        print('LONG PACE SEGMENT', rows[-1], flush=True)

    tts = IndexTTS2(model_dir=str(REPO / 'checkpoints'), cfg_path=str(REPO / 'checkpoints/config.yaml'),
        device='cuda:0', use_bf16=True, use_cuda_kernel=False, use_qwen_emo=False,
        use_deepspeed=False, use_accel=False, use_torch_compile=False)
    tts.gpt = PeftModel.from_pretrained(tts.gpt, VERSION / 'adapters/gpt').eval()
    tts.s2mel.models['cfm'] = PeftModel.from_pretrained(
        tts.s2mel.models['cfm'], VERSION / 'adapters/cfm').eval()
    token_count = lambda text: len(tts.tokenizer.encode('<|zh|> ' + text, allowed_special='all'))
    for index, (pcm, text, factor) in enumerate(zip([first[:534528], first[538938:]], first_texts, [1.1, .948])):
        add_segment(pcm, 0, index, text, factor, token_count(text), 'User-approved opening, exact original PCM')
    splitter = tts.split_text_by_tokens
    state = {'texts': [], 'index': 0, 'controls': []}

    def tracked_split(text, max_tokens, lang_prefix=''):
        state['texts'] = splitter(text, max_tokens, lang_prefix)
        return state['texts']

    def control_length(module, args, kwargs):
        # Prompt conditioning runs before text splitting and must keep its original length.
        if state['texts']:
            index = state['index']
            text = state['texts'][index]
            count = token_count(text)
            factor = .948 if index == len(state['texts'])-1 and count < 60 else 1.1
            semantic = args[0]
            kwargs = dict(kwargs)
            kwargs['ylens'] = torch.LongTensor([int(semantic.shape[1] * 1.72 * factor)]).to(semantic.device)
            state['controls'].append({'index': index, 'factor': factor, 'text_tokens': count,
                                      'target_frames': int(kwargs['ylens'].item())})
        return args, kwargs

    tts.split_text_by_tokens = tracked_split
    hook = tts.s2mel.models['length_regulator'].register_forward_pre_hook(control_length, with_kwargs=True)
    paragraph_rows = [{'paragraph': 1, 'text': paragraphs[0], 'path': 'paragraph-01.wav',
                       'duration_seconds': len(first)/sr, 'opening_pcm_identical': True}]
    shutil.copyfile(first_path, OUT / 'paragraph-01.wav')
    try:
        for paragraph_index, text in enumerate(paragraphs[1:], start=1):
            state.update(texts=[], index=0, controls=[])
            seed = 20261022 + paragraph_index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            path = OUT / f'paragraph-{paragraph_index+1:02d}.wav'
            emitted_pieces = []
            started = time.monotonic()
            generator = tts.infer(str(VERSION / 'reference.wav'), text, str(path),
                lang='ZH', use_emo_text=False, use_random=False, verbose=False,
                stream_return=True, interval_silence=200, duration_factor=1.0,
                max_text_tokens_per_segment=120, max_mel_tokens=1500,
                do_sample=True, top_p=.8, top_k=30, temperature=.8, num_beams=3)
            for emitted, chunk in enumerate(generator):
                if emitted % 2:
                    continue
                index = state['index']
                control = state['controls'][index]
                assert control['index'] == index
                pcm = chunk.to(torch.int16).numpy().reshape(-1)
                emitted_pieces.append(pcm)
                add_segment(pcm, paragraph_index, index, state['texts'][index], control['factor'],
                            control['text_tokens'], 'Native synthesis with per-segment length control')
                state['index'] += 1
            assert state['index'] == len(state['texts']) == len(state['controls'])
            audio, audio_sr = sf.read(path, dtype='int16')
            joined = np.concatenate([part for index, pcm in enumerate(emitted_pieces)
                                     for part in ([pcm] if index == 0 else [silence, pcm])])
            assert audio_sr == sr and np.array_equal(audio, joined)
            assert normalize_text(''.join(state['texts'])) == normalize_text(text)
            paragraph_rows.append({'paragraph': paragraph_index+1, 'text': text, 'path': path.name,
                'seed': seed, 'duration_seconds': len(audio)/sr, 'controls': state['controls'],
                'inference_seconds': time.monotonic()-started})
            save('synthesis.json', {'paragraphs': paragraph_rows, 'segments': rows})
            print('LONG PACE PARAGRAPH COMPLETE', paragraph_index+1, flush=True)
    finally:
        hook.remove()
    complete = np.concatenate(pieces)
    sf.write(OUT / 'story-paced.wav', complete, sr, subtype='PCM_16')
    reread, _ = sf.read(OUT / 'story-paced.wav', dtype='int16')
    assert np.array_equal(reread, complete) and np.array_equal(reread[:len(first)], first)
    assert normalize_text(''.join(row['text'] for row in rows)) == normalize_text(story['text'])
    shutil.copyfile(ROOT / 'robustness-v1-20261010' / baseline['path'], OUT / 'story-original.wav')
    del tts
    gc.collect()
    torch.cuda.empty_cache()
    import whisper
    from rapidfuzz.distance import Levenshtein
    asr = whisper.load_model('turbo', device='cuda', download_root=str(ROOT / 'asr-models'))
    for row in rows:
        result = asr.transcribe(str(OUT / row['path']), language='zh', temperature=0,
            beam_size=5, condition_on_previous_text=False, initial_prompt='以下是清晰的中文普通话语音。', fp16=True)
        expected, actual = normalize_text(row['text']), normalize_text(result['text'])
        edits = Levenshtein.editops(expected, actual)
        row.update(asr_text=result['text'].strip(), target_characters=len(expected), character_edits=len(edits),
            cer=len(edits)/max(1, len(expected)), substitutions=sum(e.tag == 'replace' for e in edits),
            deletions=sum(e.tag == 'delete' for e in edits), insertions=sum(e.tag == 'insert' for e in edits),
            characters_per_second=len(expected)/row['duration_seconds'])
        save('scoring-progress.json', rows)
        print('LONG PACE SCORED', row['paragraph'], row['segment'], row['cer'], flush=True)
    verify_version()
    metadata = {'version': manifest['version'], 'text': story['text'], 'characters': len(story['text']),
        'paragraphs': paragraph_rows, 'segments': rows, 'duration_seconds': len(complete)/sr,
        'sample_rate': sr, 'segment_count': len(rows), 'opening_pcm_identical': True,
        'selected_opening_sha256': selected_hashes[first_path.name],
        'cer': sum(row['character_edits'] for row in rows) / sum(row['target_characters'] for row in rows),
        'original_duration_seconds': baseline['duration'], 'original_cer': baseline['segment_corpus_cer'],
        'original_segment_count': len(baseline['segments']),
        'policy': 'Paragraph boundaries, 120-token budget; factor 1.1 except final segment below 60 tokens uses 0.948. Approved opening retained exactly. 200ms gaps.',
        'controls': 'Experiment-local PyTorch length-regulator pre-hook; original native frame formula; prompt length unchanged.',
        'asr': 'Whisper turbo, Chinese, temperature 0, beam 5, independently scored segments.',
        'preserved_version_verified_before_and_after': True,
        'limitations': 'New segmentation and seeds confound comparison. Short-tail rule is experimental. ASR and text rate are proxies; no human listening on new paragraphs. No new training.'}
    save('metrics.json', metadata)
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    hashes = {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in OUT.rglob('*') if p.is_file()}
    save('checksums.json', hashes)
    archive_path = ROOT / 'long-paced-v1-results.tar.gz'
    if archive_path.exists():
        raise RuntimeError('Archive already exists; choose a new archive name.')
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(OUT, arcname=OUT.name)
    print('LONG PACE COMPLETE', metadata['duration_seconds'], metadata['cer'],
          archive_path.stat().st_size, hashlib.sha256(archive_path.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
