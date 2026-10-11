import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
OUT = ROOT / 'pace-v1-20261010'
VERSION = ROOT / 'versions/mirror-joint-v1-20261010'


def verify_version():
    manifest = json.loads((VERSION / 'version.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((VERSION / name).read_bytes()).hexdigest() == expected, name
    return manifest


def main():
    import numpy as np
    import soundfile as sf

    manifest = verify_version()
    if OUT.exists():
        raise RuntimeError('Output already exists; choose a new directory.')
    OUT.mkdir()
    story = next(row for row in json.loads((ROOT / 'robustness-v1-20261010/texts.json').read_text())
                 if row['id'] == '03')
    text = story['text'].splitlines()[0]
    rows = []
    for factor in [1.0, 1.1, 1.2]:
        filename = f'pace-{factor:.2f}.wav'
        command = [sys.executable, str(ROOT / 'infer-pace-v1.py'),
                   '--repo', str(ROOT / 'repo'), '--models', str(ROOT / 'repo/checkpoints'),
                   '--version', str(VERSION), '--text', text, '--output', str(OUT / filename),
                   '--seed', '20261022', '--segment-budget', '120', '--max-mel-tokens', '1500',
                   '--duration-factor', str(factor)]
        subprocess.run(command, check=True)
        audio, sr = sf.read(OUT / filename)
        assert np.isfinite(audio).all() and len(audio) > sr
        rows.append({'duration_factor': factor, 'path': filename, 'duration_seconds': len(audio) / sr,
                     'sample_rate': sr, 'finite': True,
                     'clipping_ratio': float((np.abs(audio) >= .999).mean())})
        print('PACE GENERATED', rows[-1], flush=True)

    sys.path.insert(0, str(ROOT / 'repo'))
    import torch
    import whisper
    from rapidfuzz.distance import Levenshtein
    from webui_app.training.reward import normalize_text
    torch.set_num_threads(8)
    asr = whisper.load_model('turbo', device='cuda', download_root=str(ROOT / 'asr-models'))
    expected = normalize_text(text)
    for row in rows:
        transcription = asr.transcribe(str(OUT / row['path']), language='zh', temperature=0,
            beam_size=5, condition_on_previous_text=False, fp16=True)['text'].strip()
        row.update(asr_text=transcription,
                   cer=Levenshtein.distance(expected, normalize_text(transcription)) / len(expected),
                   duration_ratio= row['duration_seconds'] / rows[0]['duration_seconds'])
        print('PACE SCORED', row, flush=True)

    verify_version()
    metadata = {'version': manifest['version'], 'text': text, 'seed': 20261022,
                'segment_budget': 120, 'max_mel_tokens': 1500, 'rows': rows,
                'preserved_version_verified_before_and_after': True,
                'control': 'Native synthesis duration_factor; no waveform time stretching.',
                'asr': 'Whisper turbo, Chinese, beam 5, temperature 0, context disabled.',
                'limitations': 'Short excerpt comparison; no human listening assessment.'}
    (OUT / 'metrics.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    for name in ['pace-comparison.py', 'infer-pace-v1.py']:
        shutil.copyfile(ROOT / name, OUT / name)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
    (OUT / 'checksums.json').write_text(json.dumps(hashes, indent=2))
    archive_path = ROOT / 'pace-v1-results.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(OUT, arcname=OUT.name)
    print('PACE COMPLETE', archive_path, archive_path.stat().st_size,
          hashlib.sha256(archive_path.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
