"""
Merge LoRA adapter with base model to create a standalone fine-tuned model.
"""
import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from unsloth import FastVisionModel

def merge_and_save(base_model_path, adapter_path, output_path):
    print(f"Loading base model: {base_model_path}")
    print(f"Loading adapter: {adapter_path}")

    model, processor = FastVisionModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=8000,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    print("Loading LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print("✓ Merge complete!")
    print(f"Merged model size: ~7GB (full Qwen2.5-VL-7B)")

if __name__ == '__main__':
    BASE_MODEL = "/root/autodl-tmp/yao/models1_cache/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master"
    ADAPTER_DIR = "/root/autodl-tmp/yao/VDEvo/problems/vision_gen/seed_baseline_lora"
    OUTPUT_DIR = "/root/autodl-tmp/yao/VDEvo/problems/vision_gen/seed_baseline_merged"

    if len(sys.argv) > 1:
        ADAPTER_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]

    merge_and_save(BASE_MODEL, ADAPTER_DIR, OUTPUT_DIR)
