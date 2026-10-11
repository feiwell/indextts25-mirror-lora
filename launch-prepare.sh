#!/usr/bin/env bash
set -euo pipefail
cd /root/voice-training/indextts25-mirror-20261010/repo
for attempt in $(seq 1 60); do
    if .venv/bin/python -c 'import torch, whisper, transformers' >/dev/null 2>&1; then
        .venv/bin/python -u ../prepare.py
        .venv/bin/python -u ../curate.py
        exit 0
    fi
    if (( attempt % 3 == 0 )); then
        echo "Waiting for environment installation (${attempt}/60)"
    fi
    sleep 10
done
echo 'Environment did not become ready within 10 minutes'
exit 1
