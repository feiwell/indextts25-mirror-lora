#!/usr/bin/env bash
set -euo pipefail
root=/root/voice-training/indextts25-mirror-20261010
cd "$root/repo"
.venv/bin/python -u ../experiment.py score
.venv/bin/python -u ../infer-adapters.py --repo "$root/repo" --models "$root/repo/checkpoints" --results "$root" --text '今天下午三点，我们在地铁站旁边的咖啡店见面。' --variant both --output "$root/listening/reproduced/01.wav"
.venv/bin/python -u ../validate-inference.py
