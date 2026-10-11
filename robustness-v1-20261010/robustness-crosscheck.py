import gc
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
OUT = ROOT / 'robustness-v1-20261010'
sys.path.insert(0, str(ROOT / 'repo'))


def main():
    import torch
    from funasr import AutoModel
    from rapidfuzz.distance import Levenshtein
    from webui_app.training.reward import normalize_text
    torch.set_num_threads(8)
    rows = json.loads((OUT / 'metrics.json').read_text())['rows']
    assert len(rows) == 11
    model = AutoModel(model='/model/ModelScope/gongjy/SenseVoiceSmall', device='cuda:0',
                      disable_update=True, trust_remote_code=False)
    diagnostics = []
    for row in rows:
        segments = []
        for segment in row['segments']:
            result = model.generate(input=str(OUT / segment['path']), cache={}, language='zh',
                                    use_itn=False, batch_size_s=30)
            text = re.sub(r'<\|[^|]*\|>', '', result[0]['text']).strip()
            target, actual = normalize_text(segment['text']), normalize_text(text)
            ops = Levenshtein.editops(target, actual)
            segments.append({'path': segment['path'], 'asr_text': text,
                'character_edits': len(ops), 'target_characters': len(target),
                'cer': len(ops)/len(target),
                'substitutions': sum(op.tag == 'replace' for op in ops),
                'deletions': sum(op.tag == 'delete' for op in ops),
                'insertions': sum(op.tag == 'insert' for op in ops)})
        record = {'id': row['id'], 'label': row['label'], 'segments': segments,
            'segment_corpus_cer': sum(s['character_edits'] for s in segments)/sum(s['target_characters'] for s in segments)}
        diagnostics.append(record)
        print('SENSEVOICE CROSSCHECK', row['label'], row['id'], record['segment_corpus_cer'], flush=True)
    (OUT / 'asr-diagnostics.json').write_text(json.dumps({
        'asr': 'SenseVoiceSmall, Chinese, ITN disabled, independent segments', 'rows': diagnostics,
        'limitations': 'Independent ASR check only; not a human listening review.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print('CROSSCHECK COMPLETE', flush=True)
    for name in ['robustness-crosscheck.py', 'robustness-crosscheck.log']:
        shutil.copyfile(ROOT / name, OUT / name)


if __name__ == '__main__':
    main()
