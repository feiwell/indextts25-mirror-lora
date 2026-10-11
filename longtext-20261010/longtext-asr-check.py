import gc
import json
from pathlib import Path
import re
import sys

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
OUT = ROOT / 'longtext-20261010'
sys.path.insert(0, str(ROOT / 'repo'))


def main():
    import torch
    import whisper
    from rapidfuzz.distance import Levenshtein
    from webui_app.training.reward import normalize_text
    torch.set_num_threads(8)
    rows = json.loads((OUT / 'metrics.json').read_text())['rows']
    diagnostics = {'whisper_context_disabled': [], 'sensevoice_finetuned_segments': []}
    model = whisper.load_model('turbo', device='cuda', download_root=str(ROOT / 'asr-models'))
    for row in rows:
        if row['id'] != '03':
            continue
        result = model.transcribe(str(OUT / row['path']), language='zh', temperature=0,
                                  beam_size=5, initial_prompt='以下是普通话的句子。',
                                  condition_on_previous_text=False)
        expected, observed = normalize_text(row['text']), normalize_text(result['text'])
        ops = Levenshtein.editops(expected, observed)
        record = {'label': row['label'], 'asr_text': result['text'], 'cer': len(ops)/len(expected),
                  'substitutions': sum(op.tag == 'replace' for op in ops),
                  'deletions': sum(op.tag == 'delete' for op in ops),
                  'insertions': sum(op.tag == 'insert' for op in ops)}
        diagnostics['whisper_context_disabled'].append(record)
        print('NO CONTEXT', row['label'], record['cer'], flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    from funasr import AutoModel
    model = AutoModel(model='/model/ModelScope/gongjy/SenseVoiceSmall', device='cuda:0',
                      disable_update=True, trust_remote_code=False)
    row = next(r for r in rows if r['id'] == '03' and r['label'] == 'finetuned')
    for segment in row['segments']:
        result = model.generate(input=str(OUT / segment['path']), cache={}, language='zh',
                                use_itn=True, batch_size_s=30)
        text = re.sub(r'<\|[^|]*\|>', '', result[0]['text']).strip()
        expected, observed = normalize_text(segment['text']), normalize_text(text)
        edits = Levenshtein.distance(expected, observed)
        diagnostics['sensevoice_finetuned_segments'].append({'path': segment['path'], 'asr_text': text,
            'character_edits': edits, 'target_characters': len(expected), 'cer': edits/len(expected)})
    segments = diagnostics['sensevoice_finetuned_segments']
    diagnostics['sensevoice_segment_corpus_cer'] = sum(s['character_edits'] for s in segments)/sum(s['target_characters'] for s in segments)
    diagnostics['interpretation_limit'] = 'Independent ASR crosscheck; still not a human perceptual assessment.'
    (OUT / 'asr-diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print('ASR DIAGNOSTICS COMPLETE', diagnostics['sensevoice_segment_corpus_cer'], flush=True)


if __name__ == '__main__':
    main()
