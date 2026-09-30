"""
rl_train_vision_unified.py
==========================
Unified Vision GRPO RL fine-tuning for all 6 combinatorial optimization problems:
TSP, CVRP, MIS, MVC, JSSP, PFSP

Since trl's GRPOTrainer does not support vision models, this script implements
a lightweight GRPO training loop directly on top of FastVisionModel.

Based on: Vision-Language Models as Combinatorial Optimization Solvers (ICLR 2027)

Usage:
    python rl_train_vision_unified.py \
        --problem_type cvrp \
        --model_name ./output_cvrp_vision/checkpoint-5000 \
        --train_json ./data_rl/cvrp/train/train_rl.json \
        --images_dir ./data_rl/cvrp/train/images \
        --output_dir ./output_rl_cvrp
"""

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import random
import torch
import torch.nn.functional as F
import wandb
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.models.vision import process_vision_info

# Import instruction builders from prompts.py
from prompts import INSTRUCTION_BUILDERS

# Import reward functions
from rewards import (
    optimality_reward_func_cvrp,
    feasibility_reward_func_cvrp,
    # Add other reward functions as they become available
)


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

PROBLEM_CONFIGS = {
    'tsp': {
        'instruction_builder': INSTRUCTION_BUILDERS['tsp'],
        'image_pattern': lambda idx, **kw: f"rl_train_{idx:05d}_n{kw['num_nodes']}.png",
        'required_fields': ['num_nodes', 'output', 'instance'],
        'reward_funcs': None,  # TODO: Add TSP reward functions
    },
    'cvrp': {
        'instruction_builder': INSTRUCTION_BUILDERS['cvrp'],
        'image_pattern': lambda idx, **kw: f"rl_train_{idx:05d}_n{kw['num_nodes']}.png",
        'required_fields': ['num_nodes', 'vehicle_capacity', 'output', 'instance'],
        'reward_funcs': {
            'optimality': optimality_reward_func_cvrp,
            'feasibility': feasibility_reward_func_cvrp,
        },
    },
    'mis': {
        'instruction_builder': INSTRUCTION_BUILDERS['mis'],
        'image_pattern': lambda idx, **kw: f"rl_train_{idx:05d}_n{kw['num_nodes']}.png",
        'required_fields': ['num_nodes', 'output', 'instance'],
        'reward_funcs': None,  # TODO: Add MIS reward functions
    },
    'mvc': {
        'instruction_builder': INSTRUCTION_BUILDERS['mvc'],
        'image_pattern': lambda idx, **kw: f"rl_train_{idx:05d}_n{kw['num_nodes']}.png",
        'required_fields': ['num_nodes', 'output', 'instance'],
        'reward_funcs': None,  # TODO: Add MVC reward functions
    },
    'jssp': {
        'instruction_builder': INSTRUCTION_BUILDERS['jssp'],
        'image_pattern': lambda idx, **kw: f"rl_train_{idx:05d}_j{kw['n_jobs']}_m{kw['n_machines']}.png",
        'required_fields': ['n_jobs', 'n_machines', 'output', 'instance'],
        'reward_funcs': None,  # TODO: Add JSSP reward functions
    },
    'pfsp': {
        'instruction_builder': INSTRUCTION_BUILDERS['pfsp'],
        'image_pattern': lambda idx, **kw: f"rl_train_{idx:05d}_j{kw['n_jobs']}_m{kw['n_machines']}.png",
        'required_fields': ['n_jobs', 'n_machines', 'output', 'instance'],
        'reward_funcs': None,  # TODO: Add PFSP reward functions
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified Vision GRPO RL trainer for 6 CO problems'
    )

    # Problem type (NEW)
    parser.add_argument('--problem_type', type=str, required=True,
                        choices=['tsp', 'cvrp', 'mis', 'mvc', 'jssp', 'pfsp'],
                        help='Type of optimization problem')

    # Model
    parser.add_argument('--model_name', type=str, required=True,
                        help='Path to SFT checkpoint to start RL from')
    parser.add_argument('--max_seq_length', type=int, default=20000)
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--load_in_4bit', action='store_true', default=False)

    # Data
    parser.add_argument('--train_json', type=str, required=True,
                        help='RL training JSON file path')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing RL training images')

    # GRPO parameters
    parser.add_argument('--num_generations', type=int, default=4,
                        help='Completions per prompt (G in GRPO)')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='KL penalty coefficient')
    parser.add_argument('--max_new_tokens', type=int, default=2000)

    # Training
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Prompts per gradient step')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory. Default: output_rl_{problem_type}_vision')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to RL checkpoint to resume from')

    # WandB
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='WandB project name')
    parser.add_argument('--disable_wandb', action='store_true', default=False,
                        help='Disable WandB logging')

    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def get_image_path(problem_type: str, images_dir: str, idx: int, record: dict) -> str:
    """Get image path using problem-specific pattern."""
    config = PROBLEM_CONFIGS[problem_type]
    filename = config['image_pattern'](idx, **record)
    return os.path.join(images_dir, filename)


def validate_record(problem_type: str, record: dict) -> bool:
    """Validate that record has all required fields."""
    config = PROBLEM_CONFIGS[problem_type]
    for field in config['required_fields']:
        if field not in record:
            return False
    return True


def load_dataset(problem_type: str, json_path: str, images_dir: str):
    """Load RL training dataset for specified problem type."""
    with open(json_path) as f:
        records = json.load(f)

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

        # Add image path and instance data
        rec['image_path'] = img_path
        rec['instance_data'] = rec['instance']  # Store raw instance data
        rows.append(rec)

    if invalid:
        print(f"  Skipped {invalid} records (missing required fields).")
    if missing:
        print(f"  Skipped {missing} records (image not found).")
    print(f"  Loaded {len(rows)} RL training samples for {problem_type.upper()}.")

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# MESSAGE BUILDING
# ═════════════════════════════════════════════════════════════════════════════

def build_messages(problem_type: str, sample: dict):
    """Create messages for specified problem type."""
    config = PROBLEM_CONFIGS[problem_type]

    # Load image
    image = Image.open(sample["image_path"]).convert("RGB")

    # Build instruction using problem-specific builder
    instruction = config['instruction_builder'](**sample)

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": instruction},
            ],
        }
    ]


# ═════════════════════════════════════════════════════════════════════════════
# REWARD COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_rewards(problem_type: str, sample: dict, completions: list, num_generations: int, device):
    """
    Compute rewards for generated completions based on problem type.

    Returns:
        torch.Tensor: Rewards for each completion, shape (num_generations,)
    """
    config = PROBLEM_CONFIGS[problem_type]

    if config['reward_funcs'] is None:
        # Default: return zeros if reward functions not implemented
        print(f"Warning: Reward functions for {problem_type.upper()} not implemented yet. Using zero rewards.")
        return torch.zeros(num_generations, dtype=torch.float32, device=device)

    # Problem-specific reward computation
    if problem_type == 'cvrp':
        inst = sample['instance_data']
        coords = [inst[0]] * num_generations
        demands = [inst[1]] * num_generations
        capacity = [float(inst[2]) if len(inst) > 2 else float(sample['vehicle_capacity'])] * num_generations
        gt = [sample['output']] * num_generations

        r_opt = config['reward_funcs']['optimality'](completions, gt, coords, demands, capacity)
        r_feas = config['reward_funcs']['feasibility'](completions, coords, demands, capacity)
        rewards = torch.tensor(
            [ro + rf for ro, rf in zip(r_opt, r_feas)],
            dtype=torch.float32, device=device
        )
        return rewards

    # Add other problem types here as reward functions are implemented
    else:
        return torch.zeros(num_generations, dtype=torch.float32, device=device)


# ═════════════════════════════════════════════════════════════════════════════
# GRPO LOSS
# ═════════════════════════════════════════════════════════════════════════════

def grpo_loss(model, processor, samples, args, problem_type, device):
    """
    For each sample in the batch:
      1. Generate G completions
      2. Score with reward functions
      3. Compute GRPO advantage-weighted policy gradient loss
    Returns scalar loss and mean reward.
    """
    total_loss = 0.0
    total_reward = 0.0
    n_valid = 0

    for sample in samples:
        messages = build_messages(problem_type, sample)
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]

        # ── Generate G completions ──────────────────────────
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=args.num_generations,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # Decode completions (only the generated part)
        completions = processor.tokenizer.batch_decode(
            out[:, prompt_len:], skip_special_tokens=True
        )

        # ── Compute rewards ─────────────────────────────────
        rewards = compute_rewards(problem_type, sample, completions, args.num_generations, device)
        total_reward += rewards.mean().item()

        # GRPO advantage (group-normalised)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # shape (G,)

        # ── Policy gradient loss over generated tokens ──────
        # Re-run forward pass for each completion to get log-probs
        sample_loss = torch.tensor(0.0, device=device)
        for g_idx in range(args.num_generations):
            gen_ids = out[g_idx].clone()  # clone to exit inference_mode tensor
            gen_ids = gen_ids.unsqueeze(0)  # (1, T)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16):
                logits = model(
                    input_ids=gen_ids,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                ).logits  # (1, T, V)

            # Log-probs of generated tokens only
            shift_logits = logits[0, prompt_len - 1:-1]      # (completion_len, V)
            shift_labels = gen_ids[0, prompt_len:]            # (completion_len,)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                1, shift_labels.unsqueeze(1)
            ).squeeze(1)  # (completion_len,)

            # Mean log-prob weighted by advantage
            loss_g = -(adv[g_idx] * token_log_probs.mean())
            sample_loss = sample_loss + loss_g

        total_loss += sample_loss / args.num_generations
        n_valid += 1

    if n_valid == 0:
        return None, 0.0

    return total_loss / n_valid, total_reward / n_valid


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train(args):
    """Main RL training loop."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    problem_type = args.problem_type

    # Default output_dir
    if args.output_dir is None:
        args.output_dir = f"output_rl_{problem_type}_vision"

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    # Initialize WandB
    if not args.disable_wandb:
        if args.wandb_project is None:
            args.wandb_project = f"Qwen2.5-VL_{problem_type}_vision_rl"
        wandb.init(project=args.wandb_project, name=args.output_dir, config=vars(args))
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("WandB logging disabled.")

    # ── Load model ─────────────────────────────────────────
    print(f"Loading model from {args.model_name} ...")

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
    )

    # Restore offline mode
    if old_offline:
        os.environ["HF_DATASETS_OFFLINE"] = old_offline
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    FastVisionModel.for_training(model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # ── Resume from checkpoint ─────────────────────────────
    global_step = 0
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        ckpt = args.resume_from_checkpoint
        # Load LoRA weights
        from peft import set_peft_model_state_dict
        import safetensors.torch as st
        adapter_path = os.path.join(ckpt, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            state = st.load_file(adapter_path)
            set_peft_model_state_dict(model, state)
            print(f"  Loaded LoRA weights from {adapter_path}")
        # Load optimizer state
        opt_path = os.path.join(ckpt, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"  Loaded optimizer from {opt_path}")
        # Recover step count from dir name
        try:
            global_step = int(os.path.basename(ckpt).split("-")[-1])
            print(f"  Resuming from step {global_step}")
        except Exception:
            pass

    # ── Load dataset ───────────────────────────────────────
    print(f"\nLoading {problem_type.upper()} RL training dataset...")
    print(f"  JSON: {args.train_json}")
    print(f"  Images: {args.images_dir}")
    dataset = load_dataset(problem_type, args.train_json, args.images_dir)
    random.shuffle(dataset)

    # ── Training loop ──────────────────────────────────────
    print("\n" + "="*80)
    print(f"Starting RL training for {problem_type.upper()}")
    print("="*80 + "\n")

    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")
        for i in range(0, len(dataset), args.batch_size):
            batch = dataset[i: i + args.batch_size]
            if not batch:
                continue

            loss, mean_reward = grpo_loss(model, processor, batch, args, problem_type, device)
            if loss is None:
                continue

            (loss / args.gradient_accumulation_steps).backward()

            accum_idx = (global_step + 1) % args.gradient_accumulation_steps
            if accum_idx == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % args.logging_steps == 0:
                log_dict = {
                    "train/loss":        loss.item(),
                    "train/mean_reward": mean_reward,
                    "train/step":        global_step,
                }
                if not args.disable_wandb:
                    wandb.log(log_dict, step=global_step)
                print(f"  step {global_step:5d} | loss {loss.item():.4f} | reward {mean_reward:.4f}")

            if global_step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                print(f"  Saved checkpoint to {ckpt_dir}")

    # Final save
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("\n" + "="*80)
    print(f"RL training complete! Model saved to: {args.output_dir}")
    print("="*80 + "\n")

    if not args.disable_wandb:
        wandb.finish()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    train(args)
