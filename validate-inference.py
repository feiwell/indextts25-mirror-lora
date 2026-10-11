import json
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path('/root/voice-training/indextts25-mirror-20261010')
original, original_sr = sf.read(ROOT / 'listening/finetuned-best/01.wav')
reproduced, reproduced_sr = sf.read(ROOT / 'listening/reproduced/01.wav')
exact = original_sr == reproduced_sr and np.array_equal(original, reproduced)
report = {'standalone_inference_matches_delivered_exact': bool(exact)}
(ROOT / 'inference-reproduction.json').write_text(json.dumps(report, indent=2))
if not exact:
    raise RuntimeError('Standalone inference failed exact reproduction')
print('FINAL CHECK COMPLETE', report, flush=True)
