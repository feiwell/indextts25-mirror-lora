import json
from pathlib import Path
import re
import sys

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
REPO = ROOT / 'repo'
sys.path.insert(0, str(REPO))


def main():
    import torch
    from funasr import AutoModel
    from webui_app.training import dataset as DS
    from webui_app.training.reward import cer, normalize_text
    torch.set_num_threads(8)
    name = 'mirror-second-screen'
    directory = Path(DS.dir_of(name))
    items = DS.load_meta(name)
    model = AutoModel(model='/model/ModelScope/gongjy/SenseVoiceSmall', device='cuda:0',
                      disable_update=True, trust_remote_code=False)
    report = []
    accepted = []
    for item in items:
        results = model.generate(input=item.audio_abs(directory), cache={}, language='zh',
                                 use_itn=True, batch_size_s=30)
        raw = results[0]['text']
        text = re.sub(r'<\|[^|]*\|>', '', raw).strip()
        disagreement = cer(item.text, text)
        ok = disagreement <= .12 and len(normalize_text(text)) >= 5
        report.append({'id': item.id, 'whisper': item.text, 'sensevoice': text,
                       'disagreement_cer': disagreement, 'accepted': ok,
                       'policy': 'Reject disagreement > 0.12; keep original aligned Whisper text'})
        if ok:
            accepted.append(item)
        print('CROSSCHECK', report[-1], flush=True)
    (ROOT / 'transcript-crosscheck.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    final_name = 'mirror-verified-asr'
    DS.create(final_name, note='Private experiment; two-ASR agreement, no manual listening review')
    final_dir = Path(DS.dir_of(final_name))
    import shutil
    (final_dir / 'audio').mkdir(exist_ok=True)
    for item in accepted:
        shutil.copyfile(item.audio_abs(directory), final_dir / item.audio)
        item.note += '; Whisper/SenseVoice normalized CER <=0.12'
    DS.save_meta(final_name, accepted)
    split = json.loads((directory / 'split.json').read_text())
    ids = {item.id for item in accepted}
    for subset in ['train', 'val']:
        split[subset] = [key for key in split[subset] if key in ids]
    (final_dir / 'split.json').write_text(json.dumps(split, indent=2))
    manifest = json.loads((ROOT / 'curation-manifest-v2.json').read_text())
    for row in manifest:
        if row.get('split') and row['id'] not in ids:
            row.pop('split')
            row['rejections'].append('Whisper/SenseVoice transcript disagreement')
    (ROOT / 'curation-manifest-final.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    summary = {'dataset': final_name, 'train_count': len(split['train']), 'val_count': len(split['val']),
               'train_seconds': sum(u.duration for u in accepted if u.id in split['train']),
               'val_seconds': sum(u.duration for u in accepted if u.id in split['val']),
               'review': 'Automated acoustic/event screening and two-ASR agreement; no manual listening review'}
    (ROOT / 'dataset-final-summary.json').write_text(json.dumps(summary, indent=2))
    print('FINAL DATASET', summary, flush=True)
    if len(split['train']) < 20 or len(split['val']) < 3:
        raise RuntimeError('Insufficient verified data')


if __name__ == '__main__':
    main()
