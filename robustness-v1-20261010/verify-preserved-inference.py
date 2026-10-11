import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
OUT = ROOT / 'robustness-v1-20261010'


def pcm(path):
    with wave.open(str(path)) as stream:
        fmt = (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes())
        return fmt, stream.readframes(stream.getnframes())


def main():
    dest = OUT / 'preserved-version-reload.wav'
    command = [str(ROOT / 'repo/.venv/bin/python'), str(ROOT / 'infer-preserved-version.py'),
        '--repo', str(ROOT / 'repo'), '--models', str(ROOT / 'repo/checkpoints'),
        '--version', str(ROOT / 'versions/mirror-joint-v1-20261010'),
        '--text', '今天下午三点，我们在地铁站旁边的咖啡店见面。',
        '--output', str(dest), '--seed', '20261010', '--variant', 'both']
    with (OUT / 'preserved-version-reload.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    original = ROOT / 'listening/finetuned-best/01.wav'
    assert pcm(dest) == pcm(original), 'Preserved adapters did not reproduce approved first-round audio'
    record = {'version': 'mirror-joint-v1-20261010', 'both_adapters_loaded_from_preserved_directory': True,
        'original_audio': str(original), 'restored_audio': dest.relative_to(OUT).as_posix(),
        'pcm_samples_exact': True, 'original_wav_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'restored_wav_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(), 'command': command}
    (OUT / 'preserved-version-inference-verification.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    for name in ['infer-preserved-version.py', 'verify-preserved-inference.py']:
        shutil.copyfile(ROOT / name, OUT / name)
    print('PRESERVED VERSION INFERENCE VERIFIED', record['restored_wav_sha256'], flush=True)


if __name__ == '__main__':
    main()
