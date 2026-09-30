#!/root/autodl-tmp/yao/llm_env/bin/python
"""
sft_train.py for JSSP

LoRA SFT fine-tunes a base VLM (Qwen2.5-VL) on rendered JSSP images.
Based on main_train_vision_jssp.py from VisionSolver-main-jssp.
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import unsloth  # noqa: F401 -- must be first for patching
import argparse
import importlib.util
import json

import torch
from datasets import Dataset
from PIL import Image
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator

SYSTEM_PROMPT = (
    "You are an expert scheduling solver. "
    "You analyse visual Job Shop Scheduling Problem (JSSP) instances and output feasible, "
    "near-optimal schedules in the exact format requested."
)


def load_instruction_fn(code_path):
    # Load from prompt.py
    prompt_path = code_path.replace('gpt.py', 'prompt.py')
    if os.path.exists(prompt_path):
        code_path = prompt_path

    spec = importlib.util.spec_from_file_location("candidate_gpt", code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("get_instruction_v2", "get_instruction_v1", "get_instruction"):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"No get_instruction function found in {code_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='VLM SFT trainer for vision_jssp')

    p.add_argument('--model_name', type=str,
                    default='/root/autodl-tmp/yao/models1_cache/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master')
    p.add_argument('--max_seq_length', type=int, default=20000)
    p.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'])
    p.add_argument('--load_in_4bit', action='store_true', default=False)

    p.add_argument('--train_json', type=str, required=True)
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--image_prefix', type=str, default='train')
    p.add_argument('--num_train_samples', type=int, default=5000)
    p.add_argument('--code_path', type=str, required=True,
                    help='Path to candidate code file containing get_instruction function')

    p.add_argument('--lora_r', type=int, default=64)
    p.add_argument('--lora_alpha', type=int, default=64)
    p.add_argument('--bias', type=str, default='none')
    p.add_argument('--use_gradient_checkpointing', type=str, default='unsloth')
    p.add_argument('--random_state', type=int, default=42)
    p.add_argument('--finetune_vision_layers', action='store_true', default=False)
    p.add_argument('--finetune_language_layers', action='store_true', default=True)
    p.add_argument('--finetune_attention_modules', action='store_true', default=True)
    p.add_argument('--finetune_mlp_modules', action='store_true', default=True)

    p.add_argument('--per_device_train_batch_size', type=int, default=1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=8)
    p.add_argument('--warmup_steps', type=int, default=10)
    p.add_argument('--num_train_epochs', type=int, default=1)
    p.add_argument('--max_steps', type=int, default=-1, help='If >0, overrides num_train_epochs')
    p.add_argument('--learning_rate', type=float, default=2e-4)
    p.add_argument('--logging_steps', type=int, default=10)
    p.add_argument('--optim', type=str, default='adamw_8bit')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--lr_scheduler_type', type=str, default='linear')
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--output_dir', type=str, required=True)

    return p.parse_args()


def load_dataset(json_path: str, images_dir: str, image_prefix: str, max_samples: int) -> Dataset:
    with open(json_path) as f:
        records = json.load(f)

    records = records[:max_samples]
    rows, missing = [], 0
    for idx, rec in enumerate(records):
        n_jobs = rec['n_jobs']
        n_machines = rec['n_machines']
        img_path = os.path.join(images_dir, f'{image_prefix}_{idx:05d}_j{n_jobs}_m{n_machines}.png')
        if not os.path.exists(img_path):
            missing += 1
            continue
        rows.append({
            'image_path': img_path,
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'output': rec['output'],
        })

    if missing:
        print(f'  Skipped {missing} records (image not found).')
    print(f'  Loaded {len(rows)} training samples.')
    return Dataset.from_list(rows)


def make_conversation(sample: dict, instruction_fn) -> dict:
    image = Image.open(sample['image_path']).convert('RGB')
    instruction = instruction_fn(int(sample['n_jobs']), int(sample['n_machines']))
    return {
        'messages': [
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': instruction},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': sample['output']}]},
        ]
    }


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    print(f'Model:      {args.model_name}')
    print(f'Train JSON: {args.train_json}')
    print(f'Images dir: {args.images_dir}')
    print(f'Output dir: {args.output_dir}')

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias=args.bias,
        random_state=args.random_state,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    print('Loading dataset...')
    instruction_fn = load_instruction_fn(args.code_path)
    train_dataset = load_dataset(args.train_json, args.images_dir, args.image_prefix, args.num_train_samples)
    train_dataset = train_dataset.shuffle(seed=args.seed)

    collator = UnslothVisionDataCollator(
        model=model,
        processor=processor,
        formatting_func=lambda sample: make_conversation(sample, instruction_fn),
    )

    FastVisionModel.for_training(model)

    sft_config_kwargs = dict(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        output_dir=args.output_dir,
        report_to='none',
        eval_strategy='no',
        save_strategy='no',
        remove_unused_columns=False,
        dataset_text_field='',
        dataset_kwargs={'skip_prepare_dataset': True},
        max_seq_length=args.max_seq_length,
        packing=False,
    )
    if args.max_steps > 0:
        sft_config_kwargs['max_steps'] = args.max_steps
    else:
        sft_config_kwargs['num_train_epochs'] = args.num_train_epochs

    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=SFTConfig(**sft_config_kwargs),
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f'TRAIN_DONE:{args.output_dir}')


if __name__ == '__main__':
    main()
