#!/usr/bin/env bash
set -euo pipefail
cd /root/voice-training/indextts25-mirror-20261010/repo
.venv/bin/python -u ../experiment.py "$1"
