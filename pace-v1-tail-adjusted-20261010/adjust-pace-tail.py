import hashlib
import html
import json
from pathlib import Path
import shutil
import wave

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'results/pace-v1-20261010'
OUT = BASE / 'results/pace-v1-tail-adjusted-20261010'


def read_pcm(path):
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 22050)
        return wav.readframes(wav.getnframes())


def main():
    hashes = json.loads((SOURCE / 'checksums.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, name
    version = BASE / 'versions/mirror-joint-v1-20261010'
    manifest = json.loads((version / 'version.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((version / name).read_bytes()).hexdigest() == expected, name
    if OUT.exists():
        raise RuntimeError('Output already exists; choose a new directory.')
    OUT.mkdir()
    # Native segment sizes come from the saved inference log, not a timed speech cut.
    sizes = {'pace-1.10.wav': [534528, 187904], 'pace-1.00.wav': [485888, 170752]}
    silence_frames = 4410
    segments = {}
    for name, (first, second) in sizes.items():
        pcm = read_pcm(SOURCE / name)
        assert len(pcm) == (first + silence_frames + second) * 2
        assert pcm[first * 2:(first + silence_frames) * 2] == b'\0' * (silence_frames * 2)
        segments[name] = (pcm[:first * 2], pcm[(first + silence_frames) * 2:])
    original = read_pcm(SOURCE / 'pace-1.10.wav')
    corrected = segments['pace-1.10.wav'][0] + b'\0' * (silence_frames * 2) + segments['pace-1.00.wav'][1]
    destination = OUT / 'pace-tail-default.wav'
    with wave.open(str(destination), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(corrected)
    assert read_pcm(destination) == corrected
    boundary = (sizes['pace-1.10.wav'][0] + silence_frames) / 22050
    assert corrected[:int(boundary * 22050) * 2] == original[:int(boundary * 22050) * 2]
    shutil.copyfile(SOURCE / 'pace-1.10.wav', OUT / 'pace-original-slow.wav')
    data = json.loads((SOURCE / 'metrics.json').read_text(encoding='utf-8'))
    metrics = {'version': manifest['version'], 'text': data['text'],
               'original_seconds': len(original) / 44100, 'adjusted_seconds': len(corrected) / 44100,
               'tail_start_seconds': boundary, 'original_tail_seconds': 187904 / 22050,
               'adjusted_tail_seconds': 170752 / 22050, 'prefix_pcm_identical': True,
               'tail_pcm_identical_to_default_segment': True, 'inter_segment_silence_ms': 200,
               'source': 'Preserved native synthesis segments at factors 1.1 and 1.0; no time stretching.',
               'source_hashes': {name: hashes[name] for name in sizes},
               'preserved_version_verified': True,
               'limitations': 'One excerpt; boundary-level adjustment, not a general prosody fix.'}
    (OUT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    items = ''.join(f'''<section><h2>{label}</h2><p>{seconds:.1f} 秒</p>
<audio controls preload="metadata" src="{name}"></audio></section>'''
                   for label, name, seconds in [('尾段调整版', destination.name, metrics['adjusted_seconds']),
                       ('原稍慢版', 'pace-original-slow.wav', metrics['original_seconds'])])
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>V1 尾段节奏对照</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f5f7f6;color:#18231e;font-family:system-ui,"Microsoft YaHei",sans-serif;line-height:1.7;letter-spacing:0}}
header{{border-bottom:1px solid #d5dfd8;background:white}}header>div,main{{max-width:820px;margin:auto;padding:24px}}
h1{{font-size:26px;margin:0}}header p,section p{{margin:6px 0 12px;color:#53655b}}section{{padding:20px 0;border-bottom:1px solid #d5dfd8}}
h2{{font-size:19px;margin:0}}audio{{display:block;width:100%;height:54px}}article{{margin-top:28px}}@media(max-width:480px){{header>div,main{{padding:20px}}}}</style>
<header><div><h1>V1 尾段节奏对照</h1><p>社区故事 · 开头片段</p></div></header>
<main>{items}<article><h2>原文</h2><p>{html.escape(data['text'])}</p></article></main>
<script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{{if(b!==a)b.pause()}})));</script></html>'''
    (OUT / 'listening.html').write_text(page, encoding='utf-8')
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    (OUT / 'checksums.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.iterdir() if p.is_file()}, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
