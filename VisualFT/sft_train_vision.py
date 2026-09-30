"""
main_train_vision_grid.py
=========================
Vision SFT trainer for CVRP using the new grid-style images
(data_sft/train/images_grid/).

Image naming: train_{idx:05d}_n{num_nodes}.png
JSON source:  data_sft/train_cvrp-001.json

All hyperparameters follow main_train_vision_images.py exactly.
Only differences:
  - data paths point to the new grid images
  - build_instruction() describes the new visual style
    (red star depot, C{id}[demand] labels, Min Routes hint)
"""

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json

import torch
import wandb
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from PIL import Image


# ─────────────────────────────────────────────────────────────
# Prompt template  (matches new grid image style)
# ─────────────────────────────────────────────────────────────

def build_instruction(vehicle_capacity, num_nodes=None) -> str:
    cap = int(float(vehicle_capacity))
    n_customers = (int(num_nodes) - 1) if num_nodes is not None else None
    node_hint = f"There are {n_customers} customers (nodes 1 to {n_customers}). " if n_customers is not None else ""
    return (
        f"The image shows a Capacitated Vehicle Routing Problem (CVRP) instance "
        f"on a 24x24 grid. "
        f"Each blue circle is a customer node labeled C{{id}}[demand]. "
        f"The green star is the depot (node 0). "
        f"Vehicle capacity: {cap}. "
        f"{node_hint}"
        f"Assign every customer to exactly one vehicle route so that each route "
        f"starts and ends at the depot and the total demand per route does not "
        f"exceed the vehicle capacity. Minimize the total travel distance.\n\n"
        f"Provide the solution in the following format:\n"
        f"Routes: [[0, ..., 0], [0, ..., 0], ...], Objective: <total distance>"
    )


# ─────────────────────────────────────────────────────────────
# Args  (identical to main_train_vision_images.py)
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Vision SFT trainer for CVRP – grid images style'
    )

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
                        default='train_cvrp-001.json',
                        help='Training JSON filename (under data_dir)')
    parser.add_argument('--images_dir', type=str,
                        default='train/images_grid',
                        help='Sub-directory (under data_dir) holding grid images')
    parser.add_argument('--num_train_samples', type=int, default=50000)

    # LoRA  (same as main_train_vision_images.py)
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

    # Training  (same as main_train_vision_images.py)
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
    parser.add_argument('--save_total_limit', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=500)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Image filename lookup
# naming: train_{idx:05d}_n{num_nodes}.png
# ─────────────────────────────────────────────────────────────

def get_image_path(images_dir: str, idx: int, num_nodes: int) -> str:
    filename = f"train_{idx:05d}_n{num_nodes}.png"
    return os.path.join(images_dir, filename)


# ─────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────

def load_vision_dataset(json_path: str, images_dir: str, max_samples: int) -> Dataset:
    with open(json_path) as f:
        records = json.load(f)

    records = records[:max_samples]
    rows = []
    missing = 0
    for idx, rec in enumerate(records):
        num_nodes = int(rec['num_nodes'])
        img_path  = get_image_path(images_dir, idx, num_nodes)
        if not os.path.exists(img_path):
            missing += 1
            continue
        rows.append({
            "image_path":       img_path,
            "vehicle_capacity": rec["vehicle_capacity"],
            "num_nodes":        rec["num_nodes"],
            "output":           rec["output"],
        })

    if missing:
        print(f"  Skipped {missing} records (image not found).")
    print(f"  Loaded {len(rows)} vision training samples.")
    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────
# Conversation formatter
# ─────────────────────────────────────────────────────────────

def make_conversation(sample: dict) -> dict:
    image       = Image.open(sample["image_path"]).convert("RGB")
    instruction = build_instruction(sample["vehicle_capacity"], sample["num_nodes"])
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


# ─────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────

def train_model(args):
    if args.output_dir is None:
        dir_out = (
            f"output_vision_grid_alpha{args.lora_alpha}_r{args.lora_r}"
            f"_cvrp_seq{args.max_seq_length}"
            f"_b{args.per_device_train_batch_size}"
            f"_ep{args.num_train_epochs}"
        )
    else:
        dir_out = args.output_dir

    wandb.init(
        project=args.model_name.split('/')[-1] + "_cvrp_vision_grid_sft",
        name=dir_out,
    )

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        **({"cache_dir": args.cache_dir} if args.cache_dir else {}),
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
        use_rslora=args.use_rslora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    images_dir     = os.path.join(args.data_dir, args.images_dir)
    train_json_path = os.path.join(args.data_dir, args.train_json)

    print("Loading training dataset...")
    train_dataset = load_vision_dataset(train_json_path, images_dir, args.num_train_samples)
    train_dataset = train_dataset.shuffle(seed=args.seed)

    collator = UnslothVisionDataCollator(
        model=model,
        processor=processor,
        formatting_func=make_conversation,
    )

    FastVisionModel.for_training(model)

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
            report_to="wandb",
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

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    return trainer


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
