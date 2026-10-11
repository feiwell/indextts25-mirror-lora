import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
REPO = ROOT / 'repo'
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ['HF_HOME'] = str(ROOT / 'hf-home')


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def setup():
    model = REPO / 'checkpoints'
    shared = Path('/model/ModelScope/IndexTeam/IndexTTS-2.5')
    for name in ['gpt.pth', 's2mel.pth', 'codec.pth', 'feat1.pt', 'feat2.pt',
                 'wav2vec2bert_stats.pt', 'multilingual_zh_ja_yue_char_del.tiktoken']:
        target = model / name
        if not target.exists():
            target.symlink_to(shared / name)
    hf = model / 'hf_cache'
    hf.mkdir(exist_ok=True)
    links = {'w2v-bert-2.0': '/model/HuggingFace/facebook/w2v-bert-2.0',
             'bigvgan': '/model/HuggingFace/nvidia/bigvgan_v2_22khz_80band_256x',
             'campplus_cn_common.bin': '/model/ModelScope/gongjy/campplus/campplus_cn_common.pt'}
    for name, source in links.items():
        target = hf / name
        if not target.exists():
            target.symlink_to(source)
    import urllib.request
    request = urllib.request.Request('https://api.github.com/repos/SrQingChen/IndexTTS-2.5-Pro/commits/main',
                                     headers={'User-Agent': 'voice-training-experiment'})
    try:
        commit = json.load(urllib.request.urlopen(request, timeout=20))
        revision_query = {'head_at_query': commit['sha']}
    except Exception as exc:
        revision_query = {'head_at_query': None, 'query_error': str(exc)}
    provenance = {'repository': 'https://github.com/SrQingChen/IndexTTS-2.5-Pro',
                  'archive_commit': tarfile.open(ROOT / 'source-main.tar.gz').pax_headers.get('comment'),
                  'revision_query': revision_query,
                  'source_tar_sha256': hashlib.sha256((ROOT / 'source-main.tar.gz').read_bytes()).hexdigest(),
                  'source_url': 'https://www.bilibili.com/video/BV1FA4m1G7fY',
                  'cid': 1447013227, 'creator': '美御Mirror',
                  'creator_training_permission': 'not verified',
                  'use': 'private user experiment; no model publication',
                  'source_sha256': hashlib.sha256((ROOT / 'source.m4a').read_bytes()).hexdigest(),
                  'weights': {}}
    for name in ['gpt.pth', 's2mel.pth', 'codec.pth', 'config.yaml']:
        p = model / name
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        provenance['weights'][name] = {'bytes': p.stat().st_size, 'sha256': h.hexdigest()}
    save('provenance.json', provenance)
    subprocess.run(['ffmpeg', '-y', '-i', str(ROOT / 'source.m4a'), '-ac', '1', '-ar', '16000',
                    str(ROOT / 'source-16k.wav')], check=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    subprocess.run(['ffmpeg', '-y', '-i', str(ROOT / 'source.m4a'), '-ac', '1', '-ar', '22050',
                    str(ROOT / 'source-22k.wav')], check=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    print('SETUP COMPLETE', flush=True)


def transcribe():
    import torch
    import whisper
    torch.set_num_threads(8)
    result_path = ROOT / 'transcript-whisper.json'
    if result_path.exists():
        print('ASR cache exists', flush=True)
        return
    print('Loading whisper turbo', flush=True)
    model = whisper.load_model('turbo', device='cuda', download_root=str(ROOT / 'asr-models'))
    start = time.monotonic()
    result = model.transcribe(str(ROOT / 'source-16k.wav'), language='zh', task='transcribe',
                              word_timestamps=True, verbose=False, temperature=0,
                              condition_on_previous_text=False)
    result['asr_model'] = 'openai-whisper turbo 20250625'
    result['seconds'] = time.monotonic() - start
    save('transcript-whisper.json', result)
    print('ASR COMPLETE', len(result['segments']), 'segments', result['seconds'], flush=True)


if __name__ == '__main__':
    setup()
    transcribe()
