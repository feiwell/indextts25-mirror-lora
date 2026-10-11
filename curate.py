import json
import os
from pathlib import Path
import re
import time
import sys

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
REPO = ROOT / 'repo'
sys.path.insert(0, str(REPO))
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ['HF_HOME'] = str(ROOT / 'hf-home')


def metrics(audio, sr):
    import numpy as np
    from scipy.signal import fftconvolve
    frames = np.lib.stride_tricks.sliding_window_view(audio, int(sr * .04))[::int(sr * .02)].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    active = rms > max(.0004, np.quantile(rms, .9) * .1)
    frames *= np.hanning(frames.shape[1])
    corr = fftconvolve(frames, frames[:, ::-1], mode='full', axes=1)[:, frames.shape[1]-1:]
    harmonic = corr[:, int(sr / 400):int(sr / 100)].max(axis=1) / np.maximum(corr[:, 0], 1e-12)
    voiced = (harmonic > .5) & active
    return {'rms_dbfs': float(20 * np.log10(np.sqrt((audio ** 2).mean()) + 1e-9)),
            'clip_ratio': float((np.abs(audio) > .995).mean()),
            'active_ratio': float(active.mean()),
            'voiced_ratio': float(voiced.sum() / max(1, active.sum())),
            'harmonic_median': float(np.median(harmonic[active])) if active.any() else 0.}


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    from webui_app.training import dataset as DS
    torch.set_num_threads(8)
    audio, sr = sf.read(ROOT / 'source-16k.wav', dtype='float32')
    training_audio, training_sr = sf.read(ROOT / 'source-22k.wav', dtype='float32')
    transcript = json.loads((ROOT / 'transcript-whisper.json').read_text())
    # Speech segment boundaries come from word alignment, with short context margins.
    segments = []
    pending = None
    for original in transcript['segments']:
        segment = dict(original)
        segment['source_segment_ids'] = [original['id']]
        words = segment.get('words') or []
        segment['start'] = words[0]['start'] if words else segment['start']
        segment['end'] = words[-1]['end'] if words else segment['end']
        if pending is not None:
            gap = segment['start'] - pending['end']
            if 0 <= gap <= 1.2 and segment['end'] - pending['start'] <= 10:
                pending['end'] = segment['end']
                pending['text'] += '\u3002' + segment['text']
                pending['words'] = (pending.get('words') or []) + words
                pending['avg_logprob'] = min(pending['avg_logprob'], segment['avg_logprob'])
                pending['no_speech_prob'] = max(pending['no_speech_prob'], segment['no_speech_prob'])
                pending['source_segment_ids'] += segment['source_segment_ids']
                if pending['end'] - pending['start'] >= 5:
                    segments.append(pending)
                    pending = None
                continue
            segments.append(pending)
        pending = segment
    if pending is not None:
        segments.append(pending)
    candidates = []
    for segment in segments:
        words = segment.get('words') or []
        start = max(0, (words[0]['start'] if words else segment['start']) - .12)
        end = min(len(audio) / sr, (words[-1]['end'] if words else segment['end']) + .16)
        text = segment['text'].strip()
        duration = end - start
        row = {'id': f"utt_{segment['id']:05d}", 'start': start, 'end': end,
               'duration': duration, 'text': text, 'asr_avg_logprob': segment['avg_logprob'],
               'asr_no_speech_prob': segment['no_speech_prob'], 'rejections': [],
               'source_segment_ids': segment['source_segment_ids'],
               'review': 'automatic screening; no human listening review'}
        if not 1.5 <= duration <= 12:
            row['rejections'].append('duration outside 1.5..12s')
        if segment['avg_logprob'] < -.7 or segment['no_speech_prob'] > .3:
            row['rejections'].append('uncertain transcription')
        if len(re.findall(r'[\u4e00-\u9fff]', text)) < 5:
            row['rejections'].append('insufficient Chinese speech')
        if duration > .1:
            row.update(metrics(audio[int(start*sr):int(end*sr)], sr))
            if row['voiced_ratio'] < .22:
                row['rejections'].append('low voicing / excessive whisper proxy')
            if row['rms_dbfs'] < -42 or row['clip_ratio'] > .001 or row['active_ratio'] < .5:
                row['rejections'].append('level / clipping / silence')
        candidates.append(row)
    identifier = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    print('Loading audio event classifier', flush=True)
    fe = AutoFeatureExtractor.from_pretrained(identifier)
    model = ASTForAudioClassification.from_pretrained(identifier).cuda().eval()
    selected = [r for r in candidates if not r['rejections']]
    for i in range(0, len(selected), 8):
        rows = selected[i:i+8]
        inputs = fe([audio[int(r['start']*sr):int(r['end']*sr)] for r in rows],
                    sampling_rate=sr, return_tensors='pt')
        with torch.no_grad():
            probs = model(**{k: v.cuda() for k, v in inputs.items()}).logits.sigmoid().cpu()
        for row, scores in zip(rows, probs):
            labels = {model.config.id2label[k]: float(v) for k, v in enumerate(scores)}
            row['audio_events'] = sorted(labels.items(), key=lambda x: x[1], reverse=True)[:8]
            row['music_score'] = labels.get('Music', 0)
            row['whisper_score'] = labels.get('Whispering', 0)
            if labels.get('Music', 0) > .20:
                row['rejections'].append('music classifier')
            if labels.get('Whispering', 0) > .45:
                row['rejections'].append('whisper classifier')
            if max(labels.get('Laughter', 0), labels.get('Giggle', 0)) > .45:
                row['rejections'].append('laughter classifier')
        print('CLASSIFY', min(i+8, len(selected)), '/', len(selected), flush=True)
    dataset_name = 'mirror-second-screen'
    DS.create(dataset_name, note='Private experiment, source permission unverified; automatic screening')
    ds_dir = Path(DS.dir_of(dataset_name))
    (ds_dir / 'audio').mkdir(exist_ok=True)
    val_blocks = [(300, 360), (900, 960), (1500, 1560)]
    train, val, utterances, seen = [], [], [], set()
    for row in candidates:
        text_key = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', row['text'])
        if text_key in seen:
            row['rejections'].append('duplicate transcript')
        if row['rejections']:
            continue
        belongs_val = any(row['start'] >= a and row['end'] <= b for a, b in val_blocks)
        near_val = any(row['end'] >= a - 15 and row['start'] <= b + 15 for a, b in val_blocks)
        if near_val and not belongs_val:
            row['rejections'].append('validation boundary buffer')
            continue
        seen.add(text_key)
        subset = 'val' if belongs_val else 'train'
        row['split'] = subset
        (val if belongs_val else train).append(row['id'])
        path = ds_dir / 'audio' / (row['id'] + '.wav')
        clip = training_audio[int(row['start']*training_sr):int(row['end']*training_sr)]
        sf.write(path, clip, training_sr, subtype='PCM_16')
        utterances.append(DS.Utterance(id=row['id'], audio=str(path.relative_to(ds_dir)),
                                      source='BV1FA4m1G7fY', text=row['text'], asr_text=row['text'],
                                      asr_model=transcript['asr_model'], duration=row['duration'],
                                      sample_rate=22050, status='no_features',
                                      note=f"source {row['start']:.2f}..{row['end']:.2f}s; automatic review"))
    DS.save_meta(dataset_name, utterances)
    split = {'train': train, 'val': val, 'seed': 20261010, 'created_at': time.time(),
             'val_blocks_seconds': val_blocks, 'buffer_seconds': 15, 'policy': 'time block split; exact transcript deduplication'}
    (ds_dir / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    (ROOT / 'curation-manifest-v2.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'candidate_count': len(candidates), 'train_count': len(train), 'val_count': len(val),
               'train_seconds': sum(r['duration'] for r in candidates if r.get('split') == 'train'),
               'val_seconds': sum(r['duration'] for r in candidates if r.get('split') == 'val'),
               'rejected_count': sum(bool(r['rejections']) for r in candidates),
               'thresholds': {'duration_min': 1.5, 'duration_max': 12, 'active_ratio_min': .5,
                              'voiced_ratio_min': .22, 'music_max': .2, 'whisper_max': .45},
               'segmentation': 'Merge consecutive ASR segments if gap <= 1.2s and total <= 10s; close at >=5s',
               'dataset': dataset_name,
               'review_limitation': 'Automated ASR and acoustic/event filtering; manual listening and transcript verification not completed.'}
    (ROOT / 'curation-summary-v2.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    if len(train) < 20 or len(val) < 2:
        raise RuntimeError('Insufficient non-whisper screened material; review quality before training')


if __name__ == '__main__':
    main()
