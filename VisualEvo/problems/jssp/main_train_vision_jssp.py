import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import argparse
import torch
import numpy as np

# Monkey patch torch.load to always use weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    # Force weights_only=False to allow loading checkpoints with numpy arrays
    kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import os
import json
from PIL import Image


def build_instruction(n: int, m: int) -> str:
    """
    Build instruction prompt for JSSP vision model.
    Only describes the image - no text input of job details.
    """
    return (
        f"The image shows a Job Shop Scheduling Problem (JSSP) instance with {n} jobs and {m} machines. "
        f"Each row is a job (J0, J1, ...) and each column is an operation step (Op1→Op2→...). "
        f"Each cell shows the machine ID (e.g. M2) and processing time (e.g. t=56) for that operation. "
        f"Different machines are color-coded. Arrows indicate that operations within a job must be processed strictly left to right. "
        f"Each machine can process only one job at a time.\n\n"
        f"Find the schedule that minimizes the makespan.\n\n"
        f"Provide the solution in the following format:\n"
        f"1. Schedule: List the order that jobs are processed on each machine.\n"
        f"2. Makespan: The makespan of the schedule."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Vision SFT trainer for JSSP (Qwen2.5-VL)')

    # Model
    parser.add_argument('--model_name', type=str,
                        default='/root/autodl-tmp/yao/models1_cache/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master')
    parser.add_argument('--max_seq_length', type=int, default=20000)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'])
    parser.add_argument('--load_in_4bit', action='store_true', default=False)

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='./data/sft/train_scale60')
    parser.add_argument('--train_json', type=str, default='training_jssp-001.json')
    parser.add_argument('--images_dir', type=str, default='.')
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
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--save_steps', type=int, default=500)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    return parser.parse_args()


def load_jssp_dataset(data_dir: str, train_json: str, images_dir: str, max_samples: int) -> Dataset:
    json_path = os.path.join(data_dir, train_json)
    img_root = os.path.join(data_dir, images_dir)
    with open(json_path) as f:
        records = json.load(f)

    records = records[:max_samples]
    rows = []
    missing = 0
    corrupted = 0
    for idx, rec in enumerate(records):
        n, m = int(rec.get('n', rec.get('n_jobs'))), int(rec.get('m', rec.get('n_machines')))
        img_path = os.path.join(img_root, f"{idx:06d}_{n}x{m}.png")
        if not os.path.exists(img_path):
            missing += 1
            continue

        # Skip image validation to speed up loading
        # Images will be validated when actually loaded during training

        rows.append({
            'image_path': img_path,
            'n':          n,
            'm':          m,
            'output':     rec['output'],
        })

    if missing:
        print(f"  Skipped {missing} records (image not found).")
    if corrupted:
        print(f"  Skipped {corrupted} records (corrupted images).")
    print(f"  Loaded {len(rows)} JSSP training samples.")
    return Dataset.from_list(rows)


def make_conversation(sample: dict) -> dict:
    image = Image.open(sample['image_path']).convert('RGB')
    instruction = build_instruction(sample['n'], sample['m'])
    return {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text',  'text': instruction},
                ],
            },
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': sample['output']}],
            },
        ]
    }


class LossLoggingCallback(TrainerCallback):
    """Callback to save loss at every training step to a text file."""

    def __init__(self, output_file: str, log_every_step: bool = False):
        self.output_file = output_file
        self.loss_file = None
        self.log_every_step = log_every_step
        self.current_loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Open the loss file when training begins."""
        self.loss_file = open(self.output_file, 'w', buffering=1)
        self.loss_file.write("step,loss\n")

    def on_step_end(self, args, state, control, **kwargs):
        """Record loss at every step if log_every_step is True."""
        if self.log_every_step and self.current_loss is not None:
            step = state.global_step
            self.loss_file.write(f"{step},{self.current_loss}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Write loss to file at each logging step."""
        if logs is not None and 'loss' in logs:
            step = state.global_step
            loss = logs['loss']
            self.current_loss = loss
            # Always log when on_log is called
            self.loss_file.write(f"{step},{loss}\n")

    def on_train_end(self, args, state, control, **kwargs):
        """Close the loss file when training ends."""
        if self.loss_file is not None:
            self.loss_file.close()


def train_model(args):
    if args.output_dir is None:
        dir_out = (
            f"output_vision_jssp_alpha{args.lora_alpha}_r{args.lora_r}"
            f"_seq{args.max_seq_length}"
            f"_b{args.per_device_train_batch_size}"
            f"_ep{args.num_train_epochs}"
        )
    else:
        dir_out = args.output_dir

    # wandb disabled

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

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
        use_rslora=args.use_rslora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    print("Loading JSSP training dataset...")
    train_dataset = load_jssp_dataset(args.data_dir, args.train_json, args.images_dir, args.num_train_samples)
    train_dataset = train_dataset.shuffle(seed=args.seed)

    collator = UnslothVisionDataCollator(
        model=model,
        processor=processor,
        formatting_func=make_conversation,
    )

    FastVisionModel.for_training(model)

    # Setup loss logging callback
    loss_log_file = os.path.join(dir_out, 'training_losses.txt')
    loss_callback = LossLoggingCallback(loss_log_file)

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
            report_to='none',
            eval_strategy='no',
            save_total_limit=args.save_total_limit,
            save_steps=args.save_steps,
            remove_unused_columns=False,
            dataset_text_field='',
            dataset_kwargs={'skip_prepare_dataset': True},
            max_seq_length=args.max_seq_length,
            packing=False,
        ),
        callbacks=[loss_callback],
    )

    print(f"Training losses will be saved to: {loss_log_file}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    return trainer


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
