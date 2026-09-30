#!/root/autodl-tmp/yao/llm_env/bin/python
"""
merge_lora.py for JSSP

Merges a LoRA adapter with the base model to create a standalone model.

Usage:
    python merge_lora.py --base_model <base_model_path> --adapter_dir <adapter_dir> --output_dir <output_dir>
"""

import argparse
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True, help='Base model path')
    parser.add_argument('--adapter_dir', type=str, required=True, help='LoRA adapter directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for merged model')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model from {args.base_model}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    print(f"Loading LoRA adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging LoRA adapter with base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    merged_model.save_pretrained(args.output_dir)

    processor = AutoProcessor.from_pretrained(args.adapter_dir)
    processor.save_pretrained(args.output_dir)

    print("Done!")


if __name__ == '__main__':
    main()
