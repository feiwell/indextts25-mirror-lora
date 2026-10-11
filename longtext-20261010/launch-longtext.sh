#!/usr/bin/env bash
set -euo pipefail
root=/root/voice-training/indextts25-mirror-20261010
cd "$root/repo"
.venv/bin/python -u ../longtext.py synthesize
.venv/bin/python -u ../longtext.py score
.venv/bin/python -u ../longtext.py package
