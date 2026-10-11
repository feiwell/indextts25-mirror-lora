# Mirror 联合微调 V1

这是我在 IndexTTS-2.5 上做的一版小规模声音微调。底座没有改，只加了两组 LoRA：GPT 负责文本节奏和读法，CFM 负责音色和声学细节。这一版叫 V1，训练和试听都发生在 2026 年 10 月 10 日。

仓库里放的是训练记录和我生成的对照音频。原始录音、训练切片和参考片段没有公开。

试听都是原始合成结果，没有再做降噪、均衡或手工修补。同一句话使用同一条参考、同一个随机种子和同一套采样参数，所以两边的差别来自适配器，而不是后期处理。

## 听几段对比

左边是未加载适配器的原版底座，右边是 GPT 与 CFM 一起启用后的 V1。点开文件即可播放。

### 1. 约时间

今天下午三点，我们在地铁站旁边的咖啡店见面。

- 原版：[listening/baseline/01.wav](listening/baseline/01.wav)
- 训练后：[listening/finetuned-best/01.wav](listening/finetuned-best/01.wav)

### 2. 交代家里的事

冰箱里还有牛奶和面包，记得吃完早餐再出门。

- 原版：[listening/baseline/02.wav](listening/baseline/02.wav)
- 训练后：[listening/finetuned-best/02.wav](listening/finetuned-best/02.wav)

### 3. 下雨的晚上

外面开始下雨了，我把窗户关上，你先休息一会儿。

- 原版：[listening/baseline/04.wav](listening/baseline/04.wav)
- 训练后：[listening/finetuned-best/04.wav](listening/finetuned-best/04.wav)

### 4. 周末安排

这个周末我想去公园散步，然后顺路买一点水果。

- 原版：[listening/baseline/06.wav](listening/baseline/06.wav)
- 训练后：[listening/finetuned-best/06.wav](listening/finetuned-best/06.wav)

### 5. 稍长的一段日常叙述

今天早上，我比平时提前半个小时出门。街边的早餐店刚刚开门，老板正在把热腾腾的包子放进蒸笼。我买了一杯豆浆，沿着河边慢慢走到地铁站。原本以为这个时间会很拥挤，没想到车厢里还有几个空位。到了公司以后，我先整理了昨天的会议记录，再把需要确认的问题发给同事。忙完这些事情，窗外的阳光正好照在桌面上，一天就这样开始了。

- 原版：[longtext-20261010/audio/baseline/01.wav](longtext-20261010/audio/baseline/01.wav)
- 训练后：[longtext-20261010/audio/finetuned/01.wav](longtext-20261010/audio/finetuned/01.wav)

六句短句的其余对照、只开 GPT、只开 CFM，以及另一组检查点，都在 [listening.html](listening.html)。长文、语速和稳健性试听分别在 `long-paced-v1-20261010/`、`pace-v1-*` 和 `robustness-v1-20261010/`。

## 这一版做了什么

训练集很短：28 条、大约 81 秒。另外留了 7 条、大约 22 秒做验证，文本和时间都与训练集错开。六句试听文本没有出现在训练文本里。

我只训练注意力投影。两边都是 rank 8、alpha 16、dropout 0.05、学习率 5e-5、weight decay 0.02。GPT 底座用 bf16，适配器保持 fp32，并打开梯度检查点。CFM 全程 fp32。回放比例设为 0，所以它只说明这点数据上发生了什么，不能当成通用声音模型。

| | GPT | CFM |
| --- | ---: | ---: |
| 步数 | 42 | 84 |
| 轮数 | 6 | 6 |
| batch / 梯度累积 | 2 / 2 | 1 / 2 |
| 可训练参数 | 1,474,560 | 319,488 |
| 训练前验证损失 | 5.916 | 0.581 |
| 最佳验证损失 | 5.860 | 0.578 |
| 相对下降 | 0.95% | 0.56% |

硬件是一张 RTX 4090 24GB。上面的耗时只算训练循环本身：GPT 约 14 秒，CFM 约 18 秒。转写、筛选、加载底座和评估都不算在里面。

## 自动评估

六句话合计 121 个归一化字符。字符错误率用 Whisper turbo 转写后计算，温度 0、beam 5。说话人相似度是 CAMPPlus 余弦相似度，越高表示在这个嵌入空间里越接近，不等于听感更好。

| 模型 | 字符错误率 | 相对训练参考 | 相对验证语音 |
| --- | ---: | ---: | ---: |
| 原版底座 | 0.83% | 0.894 | 0.674 |
| V1，GPT + CFM | 0.00% | 0.902 | 0.695 |
| 只训练 GPT | 0.00% | 0.892 | 0.675 |
| 只训练 CFM | 0.83% | 0.901 | 0.692 |

原版底座在这六句里只有一处字符错误，V1 没有错。样本太小，我不会把这写成稳定的准确率提升。验证语音相似度有一点上升，同样不能代替人工听审。逐句时长、音量和转写在 `evaluation-metrics.json`。

我检查过适配器确实参与了计算：可训练参数有梯度，冻结参数没有。关掉两个适配器时，第一句和原版底座逐采样点相同；打开任意一个，输出就会变。

## 权重

最终适配器：

- GPT：[repo/training_runs/gpt-first-round/adapter](repo/training_runs/gpt-first-round/adapter)
- CFM：[repo/training_runs/cfm-first-round/adapter](repo/training_runs/cfm-first-round/adapter)

每个方向还留了四个检查点。比较时用的是权重哈希不同的那一组，而不是最后一次重复保存。这些文件只是 LoRA，不能替换 IndexTTS-2.5 底座，也不要覆盖 `gpt.pth` 或 `s2mel.pth`。

底座需要自行准备 IndexTTS-2.5，以及 w2v-bert-2.0、BigVGAN 和 CAMPPlus。源码对应 IndexTTS-2.5-Pro 提交 `d3439bccb061944abf745844274aa0c2edfa35a3`。已生成的音频可以直接听，不需要底座。

配置、筛选记录和训练日志在仓库根目录的 `*-training-config.json`、`*-training-report.json`、`curation-manifest-final.json` 和 `train.log`。
