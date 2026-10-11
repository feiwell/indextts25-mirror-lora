import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import tarfile
import shutil

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
REPO = ROOT / 'repo'
OUT = ROOT / 'longtext-20261010'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))
from experiment import engine, seed

TEXTS = [
    {'id': '01', 'title': '日常叙述', 'text': '今天早上，我比平时提前半个小时出门。街边的早餐店刚刚开门，老板正在把热腾腾的包子放进蒸笼。我买了一杯豆浆，沿着河边慢慢走到地铁站。原本以为这个时间会很拥挤，没想到车厢里还有几个空位。到了公司以后，我先整理了昨天的会议记录，再把需要确认的问题发给同事。忙完这些事情，窗外的阳光正好照在桌面上，一天就这样开始了。'},
    {'id': '02', 'title': '多段说明', 'text': '准备一次周末旅行，不一定要安排得特别复杂。首先，想清楚自己更喜欢安静的地方，还是热闹的街区。如果只是想休息，可以选择离家不远的小镇，留出足够的时间散步和吃饭。\n其次，提前查看天气和交通情况。遇到下雨，不妨把室外活动换成博物馆或书店；如果火车票已经售完，也可以调整出发时间，不必为了赶路把整个行程排得太满。\n整理行李的时候，把证件、充电器和常用药放在容易找到的位置。衣服够用就好，带得太多，反而会让路上变得不方便。同行的人如果有不同的想法，最好在出发之前聊一聊，决定哪些事情一起做，哪些时间可以各自安排。\n最后，给计划留一点空白。也许你会遇见一家喜欢的小店，或者发现一条原来不知道的步道。旅行的乐趣，往往就藏在这些没有提前写进清单的时刻里。回来的路上，记得检查随身物品，也给自己留一点整理心情的时间。'},
    {'id': '03', 'title': '长篇故事与对话', 'text': '傍晚的时候，林然来到那家开在街角的旧书店。门口挂着一只铜铃，她轻轻推开门，铃声就在安静的屋子里响了起来。老板正在柜台后面整理书签，抬头看见她，笑着说，今天比上次来得早一些。林然点点头，把雨伞收好，回答说，下午的事情提前做完了，想来找一本适合周末读的书。\n她沿着靠窗的书架慢慢往里面走。那里放着旅行随笔、短篇小说，还有几本已经很久没有再版的画册。一本浅绿色封面的书吸引了她的注意。翻开第一页，里面夹着一张小小的车票，日期已经模糊，只能看清终点站的名字。她把车票拿给老板看，问这是不是上一位读者忘记取走的东西。\n老板想了一会儿，说，这本书是附近一位老师送来的。他以前喜欢坐火车去不同的城市，每到一个地方，就会找一家书店，买一本书留作纪念。后来搬家，装不下那么多东西，便把一部分送到了这里。也许这张车票，就是某一次旅行留下来的。\n林然重新坐到窗边的椅子上，读了几页。书里写的是一位年轻人第一次独自出门的经历，没有惊险的冒险，也没有特别宏大的道理，只是一些关于问路、等车和认识陌生人的小事。她忽然想起，自己也曾经因为害怕走错路，放弃过一个计划了很久的行程。现在回头看，那些担心似乎并没有当时想象得那么严重。\n外面的雨渐渐停了，街上亮起一盏又一盏灯。老板给她倒了一杯温水，说，不用着急，今天可以晚一点关门。林然笑了笑，继续往下读。读到最后一章的时候，她已经决定，下个月找一个空闲的周末，去看看车票上写着的那座城市。\n离开书店之前，她买下了这本书，却把旧车票留在柜台上。她对老板说，如果那位老师以后再回来，也许还会想起这段旅程。老板把车票夹进一个透明的小袋子，认真地点了点头。走出门的时候，空气里还有雨后的凉意。林然没有立刻回家，而是沿着街道多走了一会儿。她觉得，有些变化并不需要很大的理由，只要从一个小小的决定开始就够了。'}
]


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def synthesize():
    import numpy as np
    import soundfile as sf
    import torch
    from peft import PeftModel
    save('texts.json', TEXTS)
    save('config.json', {'reference': str(ROOT / 'reference.wav'), 'seed_base': 20261020,
                        'max_mel_tokens': 1500, 'default_segment_tokens': 120,
                        'short_segment_tokens': 60, 'interval_silence_ms': 200,
                        'top_p': .8, 'top_k': 30, 'temperature': .8, 'num_beams': 3,
                        'postprocessing': 'none; upstream segment concatenation only'})
    rows = []
    for label, budget in [('baseline', 120), ('finetuned', 120), ('finetuned-short-segments', 60)]:
        tts = engine()
        if label != 'baseline':
            runs = REPO / 'training_runs'
            tts.gpt = PeftModel.from_pretrained(tts.gpt, runs / 'gpt-first-round/adapter').eval()
            tts.s2mel.models['cfm'] = PeftModel.from_pretrained(
                tts.s2mel.models['cfm'], runs / 'cfm-first-round/adapter').eval()
        original_split = tts.split_text_by_tokens
        captured = []
        def tracked_split(text, max_tokens, lang_prefix=''):
            pieces = original_split(text, max_tokens, lang_prefix)
            captured[:] = pieces
            return pieces
        tts.split_text_by_tokens = tracked_split
        for index, item in enumerate(TEXTS):
            if label == 'finetuned-short-segments' and item['id'] != '03':
                continue
            seed(20261020 + index)
            directory = OUT / 'audio' / label
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / (item['id'] + '.wav')
            segments = []
            started = time.monotonic()
            generator = tts.infer(str(ROOT / 'reference.wav'), item['text'], str(destination),
                lang='ZH', use_emo_text=False, use_random=False, verbose=False,
                stream_return=True, max_text_tokens_per_segment=budget, interval_silence=200,
                do_sample=True, top_p=.8, top_k=30, temperature=.8, num_beams=3,
                max_mel_tokens=1500)
            for emitted, chunk in enumerate(generator):
                if emitted % 2:
                    continue
                segment_index = len(segments)
                segment_path = directory / f"{item['id']}-segment-{segment_index+1:02d}.wav"
                pcm = chunk.to(torch.int16).numpy().T
                sf.write(segment_path, pcm, 22050, subtype='PCM_16')
                segments.append({'path': str(segment_path.relative_to(OUT)),
                                 'text': captured[segment_index], 'duration': len(pcm)/22050})
                print('SEGMENT COMPLETE', label, item['id'], segment_index+1, flush=True)
            audio, sr = sf.read(destination)
            if not segments or len(segments) != len(captured) or not np.isfinite(audio).all() or np.abs(audio).max() < .001:
                raise RuntimeError('Invalid long-text output')
            row = {'label': label, 'id': item['id'], 'text': item['text'], 'segments': segments,
                   'path': str(destination.relative_to(OUT)), 'characters': len(item['text']),
                   'duration': len(audio)/sr, 'inference_seconds': time.monotonic()-started,
                   'segment_budget': budget, 'seed': 20261020+index}
            rows.append(row)
            save('synthesis.json', rows)
            print('LONGTEXT COMPLETE', label, item['id'], row['duration'], flush=True)
        del tts
        gc.collect()
        torch.cuda.empty_cache()


def score():
    import numpy as np
    import soundfile as sf
    import whisper
    from rapidfuzz.distance import Levenshtein
    from webui_app.training.reward import RewardOptions, RewardScorer, normalize_text
    scorer = RewardScorer(RewardOptions(whisper_size='turbo', beam_size=5), model_dir=str(REPO / 'checkpoints'))
    scorer._asr = whisper.load_model('turbo', device='cuda', download_root=str(ROOT / 'asr-models'))
    ref = scorer.embed(str(ROOT / 'reference.wav'))
    rows = json.loads((OUT / 'synthesis.json').read_text())
    for row in rows:
        hyp = scorer.transcribe(str(OUT / row['path']))
        target, actual = normalize_text(row['text']), normalize_text(hyp)
        operations = Levenshtein.editops(target, actual)
        row.update(asr_text=hyp, target_characters=len(target), cer=len(operations)/len(target),
                   substitutions=sum(op.tag == 'replace' for op in operations),
                   deletions=sum(op.tag == 'delete' for op in operations),
                   insertions=sum(op.tag == 'insert' for op in operations))
        for segment in row['segments']:
            segment['asr_text'] = scorer.transcribe(str(OUT / segment['path']))
            expected, observed = normalize_text(segment['text']), normalize_text(segment['asr_text'])
            segment['character_edits'] = Levenshtein.distance(expected, observed)
            segment['target_characters'] = len(expected)
            segment['cer'] = segment['character_edits']/max(1,len(expected))
            segment['reference_similarity'] = scorer.cosine(ref, scorer.embed(str(OUT / segment['path'])))
        row['segment_corpus_cer'] = sum(s['character_edits'] for s in row['segments'])/sum(s['target_characters'] for s in row['segments'])
        row['first_segment_similarity'] = row['segments'][0]['reference_similarity']
        row['last_segment_similarity'] = row['segments'][-1]['reference_similarity']
        audio, sr = sf.read(OUT / row['path'])
        row['finite'] = bool(np.isfinite(audio).all())
        row['clipping_ratio'] = float((np.abs(audio)>=.999).mean())
        row['rms_dbfs'] = float(20*np.log10(np.sqrt((audio**2).mean())+1e-12))
        save('metrics.json', {'rows': rows, 'asr': 'Whisper turbo, temperature 0, beam 5, generic Chinese prompt',
                             'limitations': 'ASR text edits and CAMPPlus similarity are automated proxies; no human listening review.'})
        print('LONGTEXT SCORE', row['label'], row['id'], row['cer'], row['segment_corpus_cer'], flush=True)
    scorer.unload()


def package():
    metrics = json.loads((OUT / 'metrics.json').read_text())
    assert len(metrics['rows']) == 7 and all(r['finite'] for r in metrics['rows'])
    for name in ['longtext.py', 'launch-longtext.sh', 'provenance.json', 'reference.wav', 'longtext.log']:
        shutil.copyfile(ROOT / name, OUT / name)
    for name in ['longtext-asr-check.py', 'longtext-asr-check.log']:
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, OUT / name)
    hashes = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in OUT.rglob('*') if p.is_file() and p.name != 'checksums.json'}
    save('checksums.json', hashes)
    destination = ROOT / 'longtext-results.tar.gz'
    with tarfile.open(destination, 'w:gz') as archive:
        archive.add(OUT, arcname=OUT.name)
    print('LONGTEXT PACKAGE', destination.stat().st_size,
          hashlib.sha256(destination.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['synthesize', 'score', 'package'])
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    {'synthesize': synthesize, 'score': score, 'package': package}[args.stage]()
