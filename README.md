# Mirror 联合微调 V1 结果（2026-10-10）

这里只公开这一版的训练记录和生成示例。原始视频、训练用音频切片、参考录音 `reference.wav` 和数据集特征都没有包含在本仓库中。

详细说明见下文。生成音频在 `listening/`、`long-paced-v1-20261010/`、`longtext-20261010/`、`pace-v1-*` 和 `robustness-v1-20261010/`。

---

# IndexTTS-2.5-Pro / Mirror 云端微调结果

已完成真正的 GPT 与 CFM LoRA 训练、适配器保存重载、推理消融和基线比较。最佳联合适配器在六句新文本的自动指标上有小幅改善；样本量不足以证明听感或通用能力提升，尚未做人工听审。没有将适配器设为本地默认模型。

## 试听与权重

- [六句对比试听](listening.html)，共 30 条正式对比音频，另有 1 条关闭适配器的控制音频。
- GPT：`repo/training_runs/gpt-first-round/adapter/`
- CFM：`repo/training_runs/cfm-first-round/adapter/`
- 每个架构保留 4 个检查点目录；其中最后两个目录是同一次最终权重的重复保存。比较使用不同权重哈希的另一组检查点。
- 原始参考：`reference.wav`；选择信息见 `reference.json`。

## 数据与训练

来源：[美御Mirror / BV1FA4m1G7fY](https://www.bilibili.com/video/BV1FA4m1G7fY)，公开音频约 1621 秒。采用 Whisper turbo 时间戳、片段合并、响度/削波/活动比例筛选、AST 音频事件排除及独立 SenseVoice 文本交叉检查。没有降噪或源分离，也没有人工逐条听审。

最终训练集 28 条、80.66 秒；验证集 7 条、21.74 秒。验证取原视频的独立时间块，训练避开验证边界 15 秒，并做文本去重。六句测试文本不与保留的数据文本相同。完整筛选记录见 `curation-manifest-final.json`、`transcript-crosscheck.json` 和 `dataset-final-summary.json`。

| 项目 | GPT | CFM |
| --- | ---: | ---: |
| 优化步骤 | 42 | 84 |
| 轮数 | 6 | 6 |
| batch / 梯度累积 | 2 / 2 | 1 / 2 |
| 适配器参数 | 1,474,560 | 319,488 |
| 训练前验证损失 | 5.916263 | 0.581278 |
| 最佳验证损失 | 5.859821 | 0.578010 |
| 相对下降 | 0.95% | 0.56% |
| 训练器报告的训练循环耗时 | 13.93 秒 | 18.09 秒 |

耗时不包含依赖安装、权重加载、转写、筛选、特征提取与推理评估。硬件为 RTX 4090 24GB；Python 3.11.17、torch 2.8.0+cu128、transformers 4.52.1。共同配置：rank 8、alpha 16、dropout 0.05、学习率 5e-5、weight decay 0.02，仅训练注意力投影。GPT 底座 bf16、适配器 fp32并使用梯度检查点；CFM fp32。回放比例为 0，只应视为小数据探索实验。完整限制与参数见两份 `*-training-config.json`。

## 自动评估

每种模型生成相同的六句话，共 121 个归一化目标字符；相同训练集参考、随机种子和采样参数。音频保持原始合成输出。Whisper turbo 使用温度 0、beam 5；CER 为编辑距离除以目标字符总数。CAMPPlus 余弦相似度分别比较固定训练参考与七条验证音频的均值，数值越高表示该嵌入下越相似，不等同于人工听感。

| 模型 | 字符错误率 | 对训练参考的相似度 | 对验证音频的相似度 |
| --- | ---: | ---: | ---: |
| 原始底座 | 0.83% | 0.8941 | 0.6742 |
| GPT + CFM 最佳 | 0.00% | 0.9017 | 0.6952 |
| 仅 GPT | 0.00% | 0.8922 | 0.6747 |
| 仅 CFM | 0.83% | 0.9009 | 0.6922 |
| 另一组检查点 | 0.00% | 0.9070 | 0.6980 |

“最佳”指训练验证损失最低的适配器，不是按六句试听分数选出的模型。另一组检查点为 GPT 第 35 步、CFM 第 70 步，其验证音频相似度略高；六句测试不足以据此重新选模型。

基线只有一次字符错误，联合微调为零错误，这个差距来自很小的测试集，不能视为稳定准确率提升。逐句转写、音量、时长、削波比例和分数见 `evaluation-metrics.json`。所有 30 条音频均为有效、非静音输出。没有证明广泛文本上的自然度、情感表达、长参考适应或通用发音能力。

## 微调生效与文件验证

两种架构均测得非零适配器梯度、冻结参数无梯度，探针保存重载后参数完全一致。正式训练结束时各抽查 5 个冻结底座张量，未发现变化；这不是对所有底座张量的完整比较。关闭两种适配器时，第一句音频与基线逐采样点完全一致；启用两种或单独适配器均改变合成输出。

已核验回传包内 226 个文件的 SHA256。记录分别在 `gradient-probes.json`、`adapter-inference-control.json`、`result-checksums.json`、`local-verification.json`。原始训练日志保留了训练器自动给出的漂移建议；其中“通用能力基本无损”等提示没有被本实验验证，不作为结论。

## 复现

源码：[SrQingChen/IndexTTS-2.5-Pro](https://github.com/SrQingChen/IndexTTS-2.5-Pro)，归档对应提交 `d3439bccb061944abf745844274aa0c2edfa35a3`。本地原始源码归档在上一层 `source-main.tar.gz`，解压源码在上一层 `IndexTTS-2.5-Pro-main`。归档 SHA256：`fd0038191caf49a5789314f6f1496321c25a980aa8afea2857c1e7b42ce32687`。

底座 IndexTeam/IndexTTS-2.5 的权重通过 Pro 仓库的真实 2.5 推理/训练接口加载。共享权重没有包含在结果包中；需要另行准备相同底座及 w2v-bert-2.0、BigVGAN、CAMPPlus。禁止把适配器直接当作完整底座文件覆盖。

| 底座文件 | SHA256 |
| --- | --- |
| gpt.pth | `43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85` |
| s2mel.pth | `9b1b0003fc189c94cc349758d7ebc25f903b7eb2de4602879959cc64ce816456` |
| codec.pth | `d15cbed16a40f478438c961fb043f68dfa6353bf56c966761315db3433e9722c` |
| config.yaml | `18adf417be3e8f5e2e48e30f7420c719170a6870619436250f360d626877870e` |

使用相同源码和 `repo/uv.lock` 创建 Linux CUDA 环境。包内的 `installed-packages.json` 记录实际安装版本。恢复模型目录的依赖后，可用随包的独立推理入口运行：

```bash
python infer-adapters.py --repo /path/to/IndexTTS-2.5-Pro \
  --models /path/to/checkpoints --results /path/to/results \
  --text '今天下午三点，我们在地铁站旁边的咖啡店见面。' \
  --variant both --output /path/to/output.wav
```

`--variant baseline/gpt/cfm/both` 对应不同组合。该入口已在云端验证，联合微调第一句与交付音频相同。在不同硬件或依赖版本上不保证逐采样点复现；本地 8GB 显卡没有验证。

训练复现：包内 `prepare.py`、`curate.py`、`crosscheck.py`、`experiment.py` 使用原始云端绝对目录 `/root/voice-training/indextts25-mirror-20261010`。若迁移目录，须更改脚本里的 ROOT、prepare/crosscheck 的共享模型路径，并将 `repo/datasets/mirror-verified-asr` 恢复到相同源码仓库。原始 JSON 日志中的云端路径仅作记录，不要直接拿来打开本地音频。已有特征与音频均保留；在 repo 工作目录依次执行：

```bash
.venv/bin/python ../experiment.py probes
.venv/bin/python ../experiment.py train
.venv/bin/python ../experiment.py evaluate
.venv/bin/python ../experiment.py score
```

重新训练前应使用新的实验目录或 run_name，避免混入本轮已有检查点。若重做筛选，先把上一层 `source/BV1FA4m1G7fY.m4a` 放到 ROOT/source.m4a，再依次运行 prepare、curate、crosscheck、baseline-features。原视频音频 SHA256：`f9a4858b21dcd8048959b480a3d5725e9b72eb0e6c1a51a52adfaffa6bd29837`。

原始回传包 SHA256：`857af65ad167f73e64bf238ed3638cf50a66cdb00a892a53ff14c31d8d087a2f`。`result-checksums.json` 覆盖云端原始交付文件；这份报告、试听页面和本地校验记录在回传后生成，其哈希另见 `delivery-checksums.json`。试听页面已检查 1440px 桌面和 390px 手机布局、全部音频加载、播放进度及单播放器播放行为；见 `listening-ui-verification.json` 与两张截图。

本轮仅为私人实验，未发布模型；创作者的训练授权尚未核实。云端实例保持原状，本地默认引擎没有切换。
