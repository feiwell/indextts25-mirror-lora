import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--version', type=Path, help='Preserved version directory containing version.json and adapters/')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'both', 'gpt', 'cfm'], default='both')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--segment-budget', type=int, default=120)
    parser.add_argument('--max-mel-tokens', type=int, default=600)
    args = parser.parse_args()
    if args.segment_budget < 1 or args.max_mel_tokens < 1:
        parser.error('Generation limits must be positive')
    if args.version:
        manifest = json.loads((args.version / 'version.json').read_text(encoding='utf-8'))
        for name in ['reference.wav'] + [f'adapters/{arch}/{file}' for arch in ['gpt', 'cfm']
                for file in ['adapter_config.json', 'adapter_model.safetensors']]:
            if hashlib.sha256((args.version / name).read_bytes()).hexdigest() != manifest['sha256'][name]:
                raise RuntimeError('Preserved version checksum mismatch: ' + name)
        adapter_paths = {arch: args.version / 'adapters' / arch for arch in ['gpt', 'cfm']}
        default_reference = args.version / 'reference.wav'
    else:
        runs = args.results / 'repo/training_runs'
        adapter_paths = {arch: runs / f'{arch}-first-round/adapter' for arch in ['gpt', 'cfm']}
        default_reference = args.results / 'reference.wav'
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
    if args.variant in ['both', 'gpt']:
        tts.gpt = PeftModel.from_pretrained(tts.gpt, adapter_paths['gpt']).eval()
    if args.variant in ['both', 'cfm']:
        tts.s2mel.models['cfm'] = PeftModel.from_pretrained(
            tts.s2mel.models['cfm'], adapter_paths['cfm']).eval()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tts.infer(str(args.reference or default_reference), args.text,
              str(args.output), lang='ZH', use_emo_text=False, use_random=False,
              verbose=False, do_sample=True, top_p=.8, top_k=30, temperature=.8,
              num_beams=3, max_mel_tokens=args.max_mel_tokens,
              max_text_tokens_per_segment=args.segment_budget)


if __name__ == '__main__':
    main()
