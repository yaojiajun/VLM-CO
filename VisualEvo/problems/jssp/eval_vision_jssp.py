"""
eval_vision_jssp.py
===================
Evaluation script for JSSP vision model trained by main_train_vision_jssp.py.

Pure vision evaluation - matches the updated training script.
"""

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import re
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ─────────────────────────────────────────────────────────────
# Prompt — MUST match main_train_vision_jssp.py exactly
# ─────────────────────────────────────────────────────────────

def build_instruction(n: int, m: int) -> str:
    """
    Build instruction prompt for JSSP vision model.
    Only describes the image - no text input of job details.
    MUST match training script exactly.
    """
    return (
        f"JSSP with {n} jobs and {m} machines. "
        f"Each job has {m} operations processed sequentially on specific machines. "
        f"Each machine processes one job at a time.\n\n"
        f"Image layout:\n"
        f"- Row = Job (0 to {n-1})\n"
        f"- Column = Operation sequence (left to right)\n"
        f"- Cell = Machine ID (M#) + Processing Time (T=#)\n"
        f"- Color = Machine type\n\n"
        f"Find the schedule minimizing makespan.\n\n"
        f"Output format:\n"
        f"Schedule: [[jobs on M0], [jobs on M1], ...], Makespan: <value>"
    )


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Eval for JSSP vision model')
    p.add_argument('--model_id', type=str,
                   default='./merged_model_jssp_500k',
                   help='Path to merged model directory or LoRA adapter')
    p.add_argument('--base_model', type=str,
                   default=None,
                   help='Base model path (if using LoRA adapter)')
    p.add_argument('--eval_json', type=str,
                   default='./data/sft/eval/test.json',
                   help='Path to eval JSON file')
    p.add_argument('--images_dir', type=str,
                   default='./data/sft/eval/jsspgraph',
                   help='Path to eval images directory')
    p.add_argument('--num_samples', type=int, default=100,
                   help='Number of samples to evaluate')
    p.add_argument('--max_new_tokens', type=int, default=3000,
                   help='Maximum tokens to generate')
    p.add_argument('--save_path', type=str,
                   default='./eval_predictions_jssp.json',
                   help='Path to save predictions')
    p.add_argument('--use_lora', action='store_true',
                   help='Load as LoRA adapter instead of merged model')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Load eval data
# ─────────────────────────────────────────────────────────────

def load_eval_rows(eval_json, images_dir, num_samples):
    """Load evaluation data matching training format."""
    with open(eval_json) as f:
        data = json.load(f)
    data = data[:num_samples]

    rows = []
    missing = 0
    corrupted = 0

    for j, rec in enumerate(data):
        n = int(rec['n'])
        m = int(rec['m'])

        # Generate image filename matching training pattern
        img_path = os.path.join(images_dir, f"{j:06d}_{n}x{m}.png")

        if not os.path.exists(img_path):
            missing += 1
            if missing <= 10:
                print(f"  [warn] image not found: {img_path}")
            continue

        # Check if image is corrupted
        try:
            img = Image.open(img_path)
            img.load()
            img.close()
        except Exception as e:
            corrupted += 1
            if corrupted <= 10:
                print(f"  [warn] corrupted image {j:06d}: {e}")
            continue

        rows.append({
            'image_path': img_path,
            'n': n,
            'm': m,
            'output': rec['output'],
        })

    if missing:
        print(f"  Skipped {missing} eval samples (image not found).")
    if corrupted:
        print(f"  Skipped {corrupted} eval samples (corrupted images).")
    print(f"  Loaded {len(rows)} eval samples.")
    return rows


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def run_inference(model, processor, image_path, n, m, max_new_tokens):
    """Run inference on a single JSSP instance (pure vision)."""
    image = Image.open(image_path).convert("RGB")
    instruction = build_instruction(n, m)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": instruction},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    output = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output


# ─────────────────────────────────────────────────────────────
# Parse and validate solution
# ─────────────────────────────────────────────────────────────

def parse_solution(text):
    """Extract schedule and makespan from model output."""
    import ast

    # Extract schedule
    schedule_match = re.search(r'Schedule:\s*(\[.*?\])\s*,?\s*Makespan:', text, re.DOTALL)
    if not schedule_match:
        schedule_match = re.search(r'Schedule:\s*(\[\[.*?\]\])', text, re.DOTALL)

    if not schedule_match:
        return None, None, "no_schedule"

    try:
        schedule = ast.literal_eval(schedule_match.group(1))
    except Exception as e:
        return None, None, f"parse_error: {str(e)[:50]}"

    # Extract makespan
    makespan_match = re.search(r'Makespan:\s*([\d.]+)', text)
    if not makespan_match:
        return schedule, None, "no_makespan"

    try:
        makespan = float(makespan_match.group(1))
    except Exception:
        return schedule, None, "makespan_parse_error"

    return schedule, makespan, "ok"


def validate_jssp_solution(schedule, n, m):
    """Validate if a JSSP schedule is well-formed."""
    if not isinstance(schedule, list):
        return False, "not_a_list"

    if len(schedule) != m:
        return False, f"wrong_num_machines(got={len(schedule)},expected={m})"

    # Check if all jobs appear in schedule
    all_jobs = set()
    for machine_schedule in schedule:
        if not isinstance(machine_schedule, list):
            return False, "machine_schedule_not_list"
        all_jobs.update(machine_schedule)

    expected_jobs = set(range(n))
    if all_jobs != expected_jobs:
        return False, f"job_coverage_error(got={sorted(all_jobs)},expected={sorted(expected_jobs)})"

    return True, "ok"


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Model: {args.model_id}")
    print(f"Eval JSON: {args.eval_json}")
    print(f"Images: {args.images_dir}")
    print(f"Samples: {args.num_samples}")

    eval_rows = load_eval_rows(args.eval_json, args.images_dir, args.num_samples)

    print(f"\nLoading model from {args.model_id} ...")

    if args.use_lora and args.base_model:
        # Load with LoRA adapter
        from peft import PeftModel
        print(f"Loading base model: {args.base_model}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Loading LoRA adapter: {args.model_id}")
        model = PeftModel.from_pretrained(model, args.model_id)
        processor = AutoProcessor.from_pretrained(args.model_id)
    else:
        # Load merged model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.model_id)

    model.eval()
    print("Model loaded successfully.\n")

    predictions = []
    feasible_count = 0
    total_gap = 0.0
    gap_count = 0

    for i, row in enumerate(tqdm(eval_rows, desc="Evaluating")):
        raw = run_inference(
            model, processor,
            row['image_path'], row['n'], row['m'],
            args.max_new_tokens,
        )

        schedule, makespan, parse_status = parse_solution(raw)

        if parse_status == "ok":
            feasible, val_status = validate_jssp_solution(schedule, row['n'], row['m'])
            if feasible:
                feasible_count += 1
                status = "FEASIBLE"

                # Extract ground truth makespan
                gt_makespan_match = re.search(r'Makespan:\s*([\d.]+)', row['output'])
                if gt_makespan_match and makespan is not None:
                    gt_makespan = float(gt_makespan_match.group(1))
                    gap = ((makespan - gt_makespan) / gt_makespan) * 100
                    total_gap += gap
                    gap_count += 1
                    print(f"[{i:3d}] n={row['n']} m={row['m']} | pred={makespan:.0f} gt={gt_makespan:.0f} gap={gap:.2f}% | {status}")
                else:
                    print(f"[{i:3d}] n={row['n']} m={row['m']} | makespan={makespan} | {status}")
            else:
                status = f"INFEASIBLE({val_status})"
                print(f"[{i:3d}] n={row['n']} m={row['m']} | {status}")
        else:
            status = f"PARSE_ERROR({parse_status})"
            print(f"[{i:3d}] n={row['n']} m={row['m']} | {status}")

        predictions.append({
            'idx': i,
            'image_path': row['image_path'],
            'n': row['n'],
            'm': row['m'],
            'prediction': raw,
            'label': row['output'],
            'status': status,
            'schedule': schedule,
            'makespan': makespan,
        })

    # Save results
    with open(args.save_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\n✓ Predictions saved to {args.save_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples:      {len(eval_rows)}")
    print(f"Feasible solutions: {feasible_count} ({feasible_count/len(eval_rows)*100:.1f}%)")

    if gap_count > 0:
        avg_gap = total_gap / gap_count
        print(f"Average gap:        {avg_gap:.2f}%")
        print(f"Gap samples:        {gap_count}")
    else:
        print(f"Average gap:        N/A (no valid comparisons)")


if __name__ == "__main__":
    main()
