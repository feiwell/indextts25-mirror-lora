import argparse
import os
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'both', 'gpt', 'cfm'], default='both')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20261010)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import numpy as np
    import torch
    from peft import PeftModel
    from indextts.infer_v2_5 import IndexTTS2
    torch.set_num_threads(8)
    tts = IndexTTS2(model_dir=str(args.models), cfg_path=str(args.models / 'config.yaml'),
                    device=args.device, use_bf16=True, use_cuda_kernel=False,
                    use_qwen_emo=False, use_deepspeed=False, use_accel=False,
                    use_torch_compile=False)
    runs = args.results / 'repo/training_runs'
    if args.variant in ['both', 'gpt']:
        tts.gpt = PeftModel.from_pretrained(tts.gpt, runs / 'gpt-first-round/adapter').eval()
    if args.variant in ['both', 'cfm']:
        tts.s2mel.models['cfm'] = PeftModel.from_pretrained(
            tts.s2mel.models['cfm'], runs / 'cfm-first-round/adapter').eval()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tts.infer(str(args.reference or args.results / 'reference.wav'), args.text,
              str(args.output), lang='ZH', use_emo_text=False, use_random=False,
              verbose=False, do_sample=True, top_p=.8, top_k=30, temperature=.8,
              num_beams=3, max_mel_tokens=600)


if __name__ == '__main__':
    main()
