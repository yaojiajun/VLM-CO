#!/root/autodl-tmp/yao/llm_env/bin/python
"""
merge_lora.py

Merge a LoRA adapter into the base model for easier deployment.

Usage:
    python merge_lora.py --base_model <base> --adapter_dir <adapter> --output_dir <output>
"""

import argparse
import os

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', type=str, required=True)
    p.add_argument('--adapter_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    print(f'Loading base model from {args.base_model}...')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )

    print(f'Loading LoRA adapter from {args.adapter_dir}...')
    model = PeftModel.from_pretrained(model, args.adapter_dir)

    print('Merging LoRA weights into base model...')
    model = model.merge_and_unload()

    print(f'Saving merged model to {args.output_dir}...')
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    print('Saving processor...')
    processor = AutoProcessor.from_pretrained(args.adapter_dir)
    processor.save_pretrained(args.output_dir)

    print('Done!')


if __name__ == '__main__':
    main()
