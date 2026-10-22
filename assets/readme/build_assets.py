# -*- coding: utf-8 -*-
"""Generate the two SVG modules embedded in this repository's README.

Both are derived from real data in the repository, not drawn by hand:

* ``hero-mirror-lora.svg`` — RMS envelopes of ``listening/baseline/01.wav``
  and ``listening/finetuned-best/01.wav`` (the same sentence, same reference,
  same seed).
* ``training-curves.svg`` — validation-loss histories from
  ``gpt-training-report.json`` and ``cfm-training-report.json``.

Run from the repository root:

    python assets/readme/build_assets.py

Requires numpy. Output overwrites the two SVGs next to this script.
"""
import json
import os
import wave

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', "
        "'Microsoft YaHei', Roboto, 'Helvetica Neue', Arial, sans-serif")
MONO = ("ui-monospace, SFMono-Regular, 'Cascadia Mono', 'JetBrains Mono', "
        "Consolas, 'Liberation Mono', monospace")

PAPER = "#F7F6F1"
BORDER = "#E3DFD4"
INK = "#141A1C"
MUTED = "#6E767B"
GREEN = "#16705E"
GRAY = "#ABB2B4"


def envelope(path, buckets=96):
    with wave.open(path, "rb") as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    if sw == 2:
        a /= 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    rms = np.array([float(np.sqrt(np.mean(b * b))) for b in np.array_split(a, buckets)])
    rms /= rms.max() or 1.0
    return {"rms": rms, "sec": n / sr, "sr": sr}


def bars(rms, x0, y_center, width, height, color):
    step = width / len(rms)
    bw = max(step * 0.62, 1.2)
    out = []
    for i, v in enumerate(rms):
        h = max(v * height, 1.2)
        x = x0 + i * step + (step - bw) / 2
        out.append(
            f'<rect x="{x:.2f}" y="{y_center - h / 2:.2f}" width="{bw:.2f}" '
            f'height="{h:.2f}" rx="{min(bw / 2, 1.4):.2f}" fill="{color}"/>'
        )
    return "\n  ".join(out)


# ------------------------------------------------------------------ hero
base = envelope(os.path.join(ROOT, "listening", "baseline", "01.wav"))
v1 = envelope(os.path.join(ROOT, "listening", "finetuned-best", "01.wav"))

H = 486
hero = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {H}" width="100%" role="img"
     aria-label="indextts25-mirror-lora：IndexTTS-2.5 单说话人双 LoRA 微调记录。同一句测试语音在原版底座与 V1 下的波形对照。">
  <title>indextts25-mirror-lora — IndexTTS-2.5 单说话人联合 LoRA</title>
  <rect width="1200" height="{H}" rx="20" fill="{PAPER}"/>
  <rect x="0.5" y="0.5" width="1199" height="{H - 1}" rx="20" fill="none" stroke="{BORDER}"/>

  <text x="64" y="74" font-family="{MONO}" font-size="16" letter-spacing="2.4" fill="{GREEN}">INDEXTTS-2.5 · GPT+CFM LoRA · 2026-10-10</text>

  <text x="62" y="146" font-family="{FONT}" font-size="44" font-weight="700" letter-spacing="-0.6" fill="{INK}">81 秒语音，</text>
  <text x="62" y="200" font-family="{FONT}" font-size="44" font-weight="700" letter-spacing="-0.6" fill="{INK}">两组 LoRA，一个声音</text>
  <text x="64" y="248" font-family="{FONT}" font-size="18" fill="{MUTED}">底座一行没改，只训练注意力投影：</text>
  <text x="64" y="276" font-family="{FONT}" font-size="18" fill="{MUTED}">GPT 管节奏与读法，CFM 管音色与细节。</text>

  <text x="644" y="74" font-family="{MONO}" font-size="14" letter-spacing="1.4" fill="{MUTED}">同一句话 · 同一参考 · 种子 20261010</text>
  <text x="644" y="104" font-family="{FONT}" font-size="17" fill="{INK}">“今天下午三点，我们在地铁站旁边的咖啡店见面。”</text>

  <text x="644" y="152" font-family="{FONT}" font-size="16" font-weight="600" fill="{GRAY}">原版底座</text>
  <text x="1136" y="152" text-anchor="end" font-family="{MONO}" font-size="15" fill="{GRAY}">{base["sec"]:.2f}s</text>
  {bars(base["rms"], 644, 191, 492, 64, GRAY)}

  <text x="644" y="248" font-family="{FONT}" font-size="16" font-weight="600" fill="{GREEN}">V1 · GPT + CFM</text>
  <text x="1136" y="248" text-anchor="end" font-family="{MONO}" font-size="15" fill="{GREEN}">{v1["sec"]:.2f}s</text>
  {bars(v1["rms"], 644, 287, 492, 64, GREEN)}

  <text x="644" y="330" font-family="{FONT}" font-size="15" fill="{MUTED}">波形取自 listening/ 真实音频的 RMS 包络，不是装饰图</text>

  <line x1="64" y1="362" x2="1136" y2="362" stroke="{BORDER}" stroke-width="1"/>''']

stats = [("28 条 / 80.7 秒", "训练语料"), ("1.79 M", "可训练参数"),
         ("32 秒", "训练耗时 · RTX 4090"), ("0.83% → 0.00%", "六句字符错误率")]
x = 64
for i, (val, lab) in enumerate(stats):
    hero.append(f'  <text x="{x}" y="404" font-family="{MONO}" font-size="22" font-weight="600" fill="{INK}">{val}</text>')
    hero.append(f'  <text x="{x}" y="428" font-family="{FONT}" font-size="15" fill="{MUTED}">{lab}</text>')
    if i < 3:
        hero.append(f'  <line x1="{x + 232}" y1="382" x2="{x + 232}" y2="436" stroke="{BORDER}" stroke-width="1"/>')
    x += 268
hero.append(f'  <text x="64" y="462" font-family="{FONT}" font-size="14" fill="{MUTED}">'
            f'六句对照试听、逐句自动指标、验证清单与失败结论，都在这个仓库里。</text>')
hero.append("</svg>\n")
open(os.path.join(OUT, "hero-mirror-lora.svg"), "w", encoding="utf-8").write("\n".join(hero))


# -------------------------------------------------------- training curves
gpt = json.load(open(os.path.join(ROOT, "gpt-training-report.json"), encoding="utf-8"))
cfm = json.load(open(os.path.join(ROOT, "cfm-training-report.json"), encoding="utf-8"))


def val_points(rep):
    return [(h["step"], h["val"]) for h in rep["history"] if h.get("val") is not None]


def panel(rep, x0, title, sub, unit):
    pts = val_points(rep)
    ys = [p[1] for p in pts]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.45 or (hi * 0.01)
    ymin, ymax = lo - pad, hi + pad
    xmax = rep["steps"]
    left, width = x0 + 36, 320
    top, bottom = 132, 292

    def sx(s):
        return left + (s / xmax) * width

    def sy(v):
        return bottom - ((v - ymin) / (ymax - ymin)) * (bottom - top)

    out = [f'  <text x="{x0}" y="88" font-family="{FONT}" font-size="20" font-weight="700" fill="{INK}">{title}</text>',
           f'  <text x="{x0}" y="112" font-family="{FONT}" font-size="15" fill="{MUTED}">{sub}</text>']
    for k in range(4):
        gy = bottom - (k / 3) * (bottom - top)
        gv = ymin + (k / 3) * (ymax - ymin)
        out.append(f'  <line x1="{left}" y1="{gy:.1f}" x2="{left + width}" y2="{gy:.1f}" stroke="{BORDER}" stroke-width="1"/>')
        out.append(f'  <text x="{left + width + 10}" y="{gy + 4:.1f}" font-family="{MONO}" font-size="12" fill="{MUTED}">{gv:.{unit}f}</text>')
    out.append(f'  <line x1="{left}" y1="{sy(ys[0]):.1f}" x2="{left + width}" y2="{sy(ys[0]):.1f}" '
               f'stroke="{GRAY}" stroke-width="1.4" stroke-dasharray="5 5"/>')
    poly = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in pts)
    out.append(f'  <polyline points="{poly}" fill="none" stroke="{GREEN}" stroke-width="2.4" stroke-linejoin="round"/>')
    for s, v in pts:
        out.append(f'  <circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="3.4" fill="{PAPER}" stroke="{GREEN}" stroke-width="2"/>')
    for s in (0, xmax // 2, xmax):
        out.append(f'  <text x="{sx(s):.1f}" y="{bottom + 22}" text-anchor="middle" font-family="{MONO}" font-size="12" fill="{MUTED}">{s}</text>')
    d = (ys[-1] - ys[0]) / ys[0] * 100
    out.append(f'  <text x="{x0}" y="{bottom + 62}" font-family="{MONO}" font-size="16" fill="{INK}">'
               f'{ys[0]:.{unit}f} <tspan fill="{MUTED}">→</tspan> {ys[-1]:.{unit}f}'
               f'  <tspan fill="{GREEN}">({d:+.2f}%)</tspan></text>')
    return out


CH = 470
curves = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {CH}" width="100%" role="img"
     aria-label="两组适配器的验证损失曲线：GPT 42 步从 5.9163 降到 5.8598（-0.95%），CFM 84 步从 0.5813 降到 0.5780（-0.56%）。">
  <title>验证损失：GPT 42 步、CFM 84 步</title>
  <rect width="1200" height="{CH}" rx="20" fill="{PAPER}"/>
  <rect x="0.5" y="0.5" width="1199" height="{CH - 1}" rx="20" fill="none" stroke="{BORDER}"/>
  <text x="64" y="58" font-family="{MONO}" font-size="15" letter-spacing="2.2" fill="{GREEN}">TRAINING · 两组适配器各自的验证损失</text>''']
curves += panel(gpt, 64, "GPT 注意力投影", "42 步 / 6 轮 / 1,474,560 可训练参数", 4)
curves += panel(cfm, 648, "CFM 估计器投影", "84 步 / 6 轮 / 319,488 可训练参数", 4)
curves.append(f'  <line x1="64" y1="418" x2="1136" y2="418" stroke="{BORDER}"/>')
curves.append(f'  <text x="64" y="444" font-family="{FONT}" font-size="13" fill="{MUTED}">'
              f'两个纵轴量级不同，分别取自 gpt-training-report.json 与 cfm-training-report.json 的 history。虚线为首轮验证损失，圆点为每次评估的实际值。</text>')
curves.append("</svg>\n")
open(os.path.join(OUT, "training-curves.svg"), "w", encoding="utf-8").write("\n".join(curves))

print("written:", sorted(os.listdir(OUT)))
