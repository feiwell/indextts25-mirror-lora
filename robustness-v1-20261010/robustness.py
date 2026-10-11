import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
sys.path.insert(0, str(ROOT))
import longtext as pipeline

OUT = ROOT / 'robustness-v1-20261010'
VERSION = ROOT / 'versions/mirror-joint-v1-20261010'
TEXTS = [
    {'id': '01', 'title': '技术说明', 'text': '录制声音之前，先把麦克风固定在合适的位置，让嘴巴与收音端保持稳定的距离。输入音量不宜过高，说到最响的一句话时，波形也应该留出余量。录完以后，先检查有没有爆音、碰撞声和明显的背景音乐，再决定是否需要重新录制。不要为了让曲线看起来整齐，反复压缩每一句话的音量；过度处理可能改变原来的呼吸和语气。保存文件时，分别保留原始录音、剪辑版本和文字记录，之后排查问题会方便得多。'},
    {'id': '02', 'title': '情绪对白与问句', 'text': '真的吗？你已经拿到录取通知了？我还以为结果要等到下个星期呢！她忍不住笑了起来，又认真问道，那你什么时候出发，行李都准备好了吗？朋友停了一会儿，小声说，其实我也有点担心，到了陌生的地方，会不会什么事情都做不好。她收起笑容，回答说，担心也没关系，我们可以一件一件准备。先把今天能做的事情做好，剩下的明天再说。你不用马上变得特别勇敢。'},
    {'id': '03', 'title': '更长的新故事', 'text': '''清晨，许宁走进社区活动室的时候，里面还没有开灯。窗外的雨刚刚停，玻璃上留着几道细细的水痕。她把背包放在门边，打开靠近墙角的灯，桌上的纸箱便显出了原来的颜色。箱子里装着旧照片、几本笔记，还有一张已经折过很多次的街区地图。今天，她要和几位邻居一起整理这些东西，为周末的社区展览做准备。
最先来的邻居姓周，大家平时叫他周叔。他端着一杯热茶，站在门口看了看，问道，这么多材料，真的能在两天里整理完吗？许宁笑着说，先把能确定的部分挑出来，剩下的可以留一个位置，请来看展览的人帮忙补充。周叔点点头，放下茶杯，从箱子里拿出一张照片。照片上是一条窄窄的小路，两边没有现在这么多商店，只有几棵树和一间修自行车的小铺。
周叔说，他小时候每天从这条路去学校，走到拐角，总会听见打气筒的声音。修车的师傅很少抬头说话，但看见孩子的轮胎瘪了，常常会招手让他们停下来。许宁把这段话记在纸上，又问，您还记得那位师傅的名字吗？周叔想了很久，摇摇头，说，大家都喊他师傅，好像没有谁认真问过名字。他说完以后，把照片放回桌上，神情有一点遗憾。
过了一会儿，隔壁楼的阿姨也来了。她带着一个透明的文件袋，里面是几封旧信和一张手写的菜谱。她解释说，这些东西看起来不算特别珍贵，可是当年搬到这里的时候，附近的邻居经常照着这张菜谱一起做饭。谁家的孩子放学早，就先到有人在家的那一户等着。那时候大家都忙，门口却总能找到一个愿意帮忙的人。
许宁听着这些故事，原来写好的展览标题忽然显得太大了。她原本想介绍街区的发展，准备按年代排出一条长长的时间线。现在她发现，真正让人停下来看的，可能是一张普通的照片，或是一句只在某个家庭里记得的话。她把时间线挪到旁边，另取一张白纸，写下照片中的路、门口的饭香和放学后的等待，准备把这些小事分别整理出来。
上午快结束的时候，一个背着书包的男孩在门口探了探头。他说老师布置了一个作业，要采访家附近的老人，了解一件过去发生的事。周叔把椅子往旁边挪了挪，请他坐下。男孩打开本子，第一句话就问，您觉得以前的生活更好，还是现在更好？屋子里安静了一瞬，周叔笑了，说，这个问题太难了，以前有以前的难处，现在也有现在的不方便，不如先问问我们当时怎么过一天。
男孩想了想，把问题改成了，您小时候最喜欢一天里的哪个时候？周叔说，是夏天吃过晚饭以后，大家搬着小凳子坐到门外。有的人聊工作，有的人修东西，孩子在旁边跑来跑去。街灯没有那么亮，可是走到哪一段，都知道门后住着谁。男孩一边听一边写，偶尔抬头追问。许宁没有打断他们，只把窗户开了一点，让屋子里的空气更清爽一些。
下午，整理工作遇到了一个小问题。两张照片的背面写着不同的年份，画面里却出现了同一块刚换好的招牌。许宁没有急着把其中一张判成错误，而是把两张并排放好，贴上需要核实的纸条。阿姨说，也许写日期的人记错了，也许照片洗出来的时候已经过了一年。周叔建议，等隔壁修表店的老板来看看，他对那块招牌应该还有印象。
天色慢慢暗下来，几张桌子终于整理出空隙。许宁把确定的信息贴到展板上，又留了几个空白位置。男孩临走前看着那些空白，问，是不是还有很多事情不知道？许宁回答说，是啊，这也是展览的一部分。我们不一定能把每个名字和日期都找回来，但可以把已经知道的认真保存，也让后来的人有机会补上新的线索。男孩点点头，说周末会带奶奶一起来。
锁门之前，许宁又看了一眼最早拿出来的街区地图。上面有几条路已经改了名字，还有一些小店早就不在了。可是今天听到的声音，让那些地方重新有了具体的样子。她没有再改展览的标题，只在下面加了一行很小的字，写着这些故事还在继续。走出活动室的时候，路边有人正收起雨伞。许宁向对方点了点头，沿着熟悉的小路慢慢往家走。'''},
    {'id': '04', 'title': '数字与时间播报（中文读法）', 'text': '各位乘客请注意，原定于上午九点四十五分出发的列车，预计晚点二十分钟。请前往二号候车区，留意屏幕上的最新通知。今天的室外温度为十二摄氏度，傍晚可能降到七摄氏度。需要办理改签的旅客，可以到三号窗口咨询。每张车票的服务费为十八元五角，办理完成后请核对姓名、日期和座位信息。预计到达时间为下午一点零五分，我们会在行程恢复后再次广播。'},
    {'id': '05', 'title': '多音字与连续问句', 'text': '银行行长今天要参加行业会议，还要检查仓库里两行货架的摆放情况。重复检查并不意味着每次都要重新开始，先分清轻重，再调整处理的顺序。你觉得这个方案合适吗？如果时间不够，我们可以延期吗？如果只改其中一部分，结果会不会更好？请想清楚以后再回答，不用着急。'}
]


def verify_version():
    manifest = json.loads((VERSION / 'version.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((VERSION / name).read_bytes()).hexdigest() == expected, name
    for arch in ['gpt', 'cfm']:
        for name in ['adapter_config.json', 'adapter_model.safetensors']:
            preserved = VERSION / 'adapters' / arch / name
            active = pipeline.REPO / 'training_runs' / f'{arch}-first-round/adapter' / name
            assert preserved.read_bytes() == active.read_bytes(), str(active)
    return manifest


def score():
    import numpy as np
    import soundfile as sf
    import whisper
    from rapidfuzz.distance import Levenshtein
    from webui_app.training.reward import RewardOptions, RewardScorer, normalize_text
    scorer = RewardScorer(RewardOptions(whisper_size='turbo', beam_size=5), model_dir=str(pipeline.REPO / 'checkpoints'))
    scorer._asr = whisper.load_model('turbo', device='cuda', download_root=str(ROOT / 'asr-models'))
    ref = scorer.embed(str(ROOT / 'reference.wav'))
    rows = json.loads((OUT / 'synthesis.json').read_text())
    for row in rows:
        for segment in row['segments']:
            transcription = scorer._asr.transcribe(str(OUT / segment['path']), language='zh',
                temperature=0, beam_size=5, condition_on_previous_text=False,
                initial_prompt='以下是清晰的中文普通话语音。', fp16=True)['text'].strip()
            target, actual = normalize_text(segment['text']), normalize_text(transcription)
            ops = Levenshtein.editops(target, actual)
            segment.update(asr_text=transcription, target_characters=len(target), character_edits=len(ops),
                cer=len(ops)/max(1, len(target)),
                substitutions=sum(op.tag == 'replace' for op in ops),
                deletions=sum(op.tag == 'delete' for op in ops),
                insertions=sum(op.tag == 'insert' for op in ops),
                reference_similarity=scorer.cosine(ref, scorer.embed(str(OUT / segment['path']))))
        expected = normalize_text(row['text'])
        actual = normalize_text(''.join(s['asr_text'] for s in row['segments']))
        ops = Levenshtein.editops(expected, actual)
        audio, sr = sf.read(OUT / row['path'])
        row.update(asr_text=''.join(s['asr_text'] for s in row['segments']),
            cer=len(ops)/len(expected),
            segment_corpus_cer=sum(s['character_edits'] for s in row['segments'])/sum(s['target_characters'] for s in row['segments']),
            substitutions=sum(op.tag == 'replace' for op in ops), deletions=sum(op.tag == 'delete' for op in ops),
            insertions=sum(op.tag == 'insert' for op in ops),
            first_segment_similarity=row['segments'][0]['reference_similarity'],
            last_segment_similarity=row['segments'][-1]['reference_similarity'],
            finite=bool(np.isfinite(audio).all()), clipping_ratio=float((np.abs(audio)>=.999).mean()),
            rms_dbfs=float(20*np.log10(np.sqrt((audio**2).mean())+1e-12)))
        pipeline.save('metrics.json', {'rows': rows,
            'asr': 'Whisper turbo, Chinese, beam 5, temperature 0, independent segments, context disabled',
            'limitations': 'Text-driven expression only; no forced emotion control or human listening. Chinese spoken numbers; not digit normalization testing.'})
        print('ROBUSTNESS SCORE', row['label'], row['id'], row['cer'], row['segment_corpus_cer'], flush=True)
    scorer.unload()


def package():
    manifest = verify_version()
    rows = json.loads((OUT / 'metrics.json').read_text())['rows']
    assert len(rows) == 11 and all(r['finite'] for r in rows)
    pipeline.save('preserved-version.json', manifest)
    for name in ['robustness.py', 'longtext.py', 'experiment.py', 'provenance.json', 'reference.wav', 'robustness.log']:
        shutil.copyfile(ROOT / name, OUT / name)
    hashes = {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in OUT.rglob('*') if p.is_file() and p.name != 'checksums.json'}
    pipeline.save('checksums.json', hashes)
    dest = ROOT / 'robustness-v1-results.tar.gz'
    with tarfile.open(dest, 'w:gz') as archive:
        archive.add(OUT, arcname=OUT.name)
    print('ROBUSTNESS PACKAGE', dest.stat().st_size, hashlib.sha256(dest.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['all', 'package'])
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    verify_version()
    pipeline.OUT = OUT
    pipeline.TEXTS = TEXTS
    if args.stage == 'all':
        if OUT.exists():
            raise RuntimeError('Output directory exists; preserve it and choose a new directory.')
        pipeline.synthesize()
        score()
    package()
