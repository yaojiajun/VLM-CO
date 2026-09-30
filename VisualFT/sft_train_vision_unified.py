"""
sft_train_vision_unified.py
============================
Unified Vision SFT trainer for all 6 combinatorial optimization problems:
TSP, CVRP, MIS, MVC, JSSP, PFSP

Based on: Vision-Language Models as Combinatorial Optimization Solvers (ICLR 2027)

Usage:
    python sft_train_vision_unified.py \
        --problem_type tsp \
        --data_dir ./data_sft/tsp \
        --train_json train_tsp.json \
        --images_dir train/images \
        --num_train_samples 5000

Supports all 6 problems through configuration-driven approach.
"""

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
from typing import Dict, Callable, List, Any

import torch
import wandb
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from PIL import Image

# Import instruction builders from prompts.py
from prompts import INSTRUCTION_BUILDERS


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

PROBLEM_CONFIGS = {
    'tsp': {
        'instruction_builder': INSTRUCTION_BUILDERS['tsp'],
        'image_pattern': lambda idx, **kw: f"train_{idx:05d}_n{kw['num_nodes']}.png",
        'required_fields': ['num_nodes', 'output'],
        'optional_fields': [],
    },
    'cvrp': {
        'instruction_builder': INSTRUCTION_BUILDERS['cvrp'],
        'image_pattern': lambda idx, **kw: f"train_{idx:05d}_n{kw['num_nodes']}.png",
        'required_fields': ['num_nodes', 'vehicle_capacity', 'output'],
        'optional_fields': [],
    },
    'mis': {
        'instruction_builder': INSTRUCTION_BUILDERS['mis'],
        'image_pattern': lambda idx, **kw: f"train_{idx:05d}_n{kw['num_nodes']}.png",
        'required_fields': ['num_nodes', 'output'],
        'optional_fields': [],
    },
    'mvc': {
        'instruction_builder': INSTRUCTION_BUILDERS['mvc'],
        'image_pattern': lambda idx, **kw: f"train_{idx:05d}_n{kw['num_nodes']}.png",
        'required_fields': ['num_nodes', 'output'],
        'optional_fields': [],
    },
    'jssp': {
        'instruction_builder': INSTRUCTION_BUILDERS['jssp'],
        'image_pattern': lambda idx, **kw: f"train_{idx:05d}_j{kw['n_jobs']}_m{kw['n_machines']}.png",
        'required_fields': ['n_jobs', 'n_machines', 'output'],
        'optional_fields': [],
    },
    'pfsp': {
        'instruction_builder': INSTRUCTION_BUILDERS['pfsp'],
        'image_pattern': lambda idx, **kw: f"train_{idx:05d}_j{kw['n_jobs']}_m{kw['n_machines']}.png",
        'required_fields': ['n_jobs', 'n_machines', 'output'],
        'optional_fields': [],
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Unified Vision SFT trainer for 6 CO problems'
    )

    # Problem type (NEW)
    parser.add_argument('--problem_type', type=str, required=True,
                        choices=['tsp', 'cvrp', 'mis', 'mvc', 'jssp', 'pfsp'],
                        help='Type of optimization problem')

    # Model
    parser.add_argument('--model_name', type=str,
                        default='/root/autodl-tmp/yao/models1/Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--max_seq_length', type=int, default=20000)
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--load_in_4bit', action='store_true', default=False)

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='./data_sft',
                        help='Root data directory')
    parser.add_argument('--train_json', type=str,
                        default=None,
                        help='Training JSON filename (under data_dir). Default: train_{problem_type}.json')
    parser.add_argument('--images_dir', type=str,
                        default='train/images',
                        help='Sub-directory (under data_dir) holding images')
    parser.add_argument('--num_train_samples', type=int, default=500000)

    # LoRA
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--bias', type=str, default='lora_only',
                        choices=['none', 'all', 'lora_only'])
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--use_rslora', action='store_true', default=False)
    parser.add_argument('--finetune_vision_layers', action='store_true', default=False)
    parser.add_argument('--finetune_language_layers', action='store_true', default=True)
    parser.add_argument('--finetune_attention_modules', action='store_true', default=True)
    parser.add_argument('--finetune_mlp_modules', action='store_true', default=True)

    # Training
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--optim', type=str, default='adamw_8bit')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_total_limit', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=1000)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)

    # WandB
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='WandB project name. Default: {model}_{problem}_vision_sft')
    parser.add_argument('--disable_wandb', action='store_true', default=False,
                        help='Disable WandB logging')

    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def get_image_path(problem_type: str, images_dir: str, idx: int, record: Dict[str, Any]) -> str:
    """Get image path using problem-specific pattern."""
    config = PROBLEM_CONFIGS[problem_type]
    filename = config['image_pattern'](idx, **record)
    return os.path.join(images_dir, filename)


def validate_record(problem_type: str, record: Dict[str, Any]) -> bool:
    """Validate that record has all required fields."""
    config = PROBLEM_CONFIGS[problem_type]
    for field in config['required_fields']:
        if field not in record:
            return False
    return True


def load_vision_dataset(
    problem_type: str,
    json_path: str,
    images_dir: str,
    max_samples: int
) -> Dataset:
    """Load vision dataset for specified problem type."""
    with open(json_path) as f:
        records = json.load(f)

    records = records[:max_samples]
    rows = []
    missing = 0
    invalid = 0

    for idx, rec in enumerate(records):
        # Validate record
        if not validate_record(problem_type, rec):
            invalid += 1
            continue

        # Convert fields to appropriate types
        if 'num_nodes' in rec:
            rec['num_nodes'] = int(rec['num_nodes'])
        if 'n_jobs' in rec:
            rec['n_jobs'] = int(rec['n_jobs'])
        if 'n_machines' in rec:
            rec['n_machines'] = int(rec['n_machines'])

        # Get image path
        img_path = get_image_path(problem_type, images_dir, idx, rec)
        if not os.path.exists(img_path):
            missing += 1
            continue

        # Add image path to record
        rec['image_path'] = img_path
        rows.append(rec)

    if invalid:
        print(f"  Skipped {invalid} records (missing required fields).")
    if missing:
        print(f"  Skipped {missing} records (image not found).")
    print(f"  Loaded {len(rows)} vision training samples for {problem_type.upper()}.")

    return Dataset.from_list(rows)


# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATION FORMATTER
# ═════════════════════════════════════════════════════════════════════════════

def make_conversation(problem_type: str, sample: dict) -> dict:
    """Create conversation format for training."""
    config = PROBLEM_CONFIGS[problem_type]

    # Load image
    image = Image.open(sample["image_path"]).convert("RGB")

    # Build instruction using problem-specific builder
    instruction = config['instruction_builder'](**sample)

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["output"]}],
            },
        ]
    }


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_model(args):
    """Main training function."""
    problem_type = args.problem_type

    # Default train_json if not specified
    if args.train_json is None:
        args.train_json = f"train_{problem_type}.json"

    # Default output_dir
    if args.output_dir is None:
        dir_out = (
            f"output_{problem_type}_vision"
            f"_alpha{args.lora_alpha}_r{args.lora_r}"
            f"_seq{args.max_seq_length}"
            f"_b{args.per_device_train_batch_size}"
            f"_ep{args.num_train_epochs}"
        )
    else:
        dir_out = args.output_dir

    # Default WandB project
    if args.wandb_project is None:
        model_short = args.model_name.split('/')[-1]
        args.wandb_project = f"{model_short}_{problem_type}_vision_sft"

    # Initialize WandB (unless disabled)
    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=dir_out,
            config=vars(args),
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("WandB logging disabled.")

    # Model dtype
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    # Load model
    print(f"Loading model: {args.model_name}")

    # Temporarily disable offline mode for local model loading
    old_offline = os.environ.get("HF_DATASETS_OFFLINE")
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        **({"cache_dir": args.cache_dir} if args.cache_dir else {}),
    )

    # Restore offline mode
    if old_offline:
        os.environ["HF_DATASETS_OFFLINE"] = old_offline
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Apply LoRA
    print("Applying LoRA adapters...")
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
        use_rslora=args.use_rslora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    # Prepare data paths
    images_dir = os.path.join(args.data_dir, args.images_dir)
    train_json_path = os.path.join(args.data_dir, args.train_json)

    print(f"\nLoading {problem_type.upper()} training dataset...")
    print(f"  JSON: {train_json_path}")
    print(f"  Images: {images_dir}")

    # Load dataset
    train_dataset = load_vision_dataset(
        problem_type,
        train_json_path,
        images_dir,
        args.num_train_samples
    )
    train_dataset = train_dataset.shuffle(seed=args.seed)

    # Create data collator with problem-specific formatter
    collator = UnslothVisionDataCollator(
        model=model,
        processor=processor,
        formatting_func=lambda sample: make_conversation(problem_type, sample),
    )

    # Prepare for training
    FastVisionModel.for_training(model)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=dir_out,
            report_to="none" if args.disable_wandb else "wandb",
            eval_strategy="no",
            save_total_limit=args.save_total_limit,
            save_steps=args.save_steps,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=args.max_seq_length,
            packing=False,
        ),
    )

    # Train
    print("\n" + "="*80)
    print(f"Starting training for {problem_type.upper()}")
    print("="*80 + "\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("\n" + "="*80)
    print(f"Training complete! Model saved to: {dir_out}")
    print("="*80 + "\n")

    return trainer


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
