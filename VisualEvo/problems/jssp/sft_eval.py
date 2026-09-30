#!/root/autodl-tmp/yao/llm_env/bin/python
"""
sft_eval.py for JSSP

Runs inference with a LoRA-fine-tuned VLM on JSSP eval instances.
Based on eval_vision_jssp.py from VisionSolver-main-jssp.
"""

import argparse
import importlib.util
import json
import os
import re
import sys

import torch
import numpy as np

# Import JSSP utilities from VisionSolver-main
sys.path.insert(0, '/root/autodl-tmp/yao/VisionSolver-main-jssp')
from Envs.JSSPEnv.JSSPEnv import get_makespan


def load_instruction_template(code_path):
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', type=str,
                    default='/root/autodl-tmp/yao/models1_cache/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master')
    p.add_argument('--adapter_dir', type=str, required=True)
    p.add_argument('--eval_json', type=str, required=True)
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--image_prefix', type=str, default='eval')
    p.add_argument('--num_samples', type=int, default=50)
    p.add_argument('--max_new_tokens', type=int, default=1000)
    p.add_argument('--infeasible_penalty', type=float, default=1.0,
                    help='Penalty for infeasible solutions')
    p.add_argument('--code_path', type=str, required=True,
                    help='Path to candidate code file containing get_instruction function')
    return p.parse_args()


def load_eval_rows(eval_json, images_dir, image_prefix, num_samples):
    with open(eval_json) as f:
        data = json.load(f)
    rows = []
    for j, rec in enumerate(data[:num_samples]):
        n_jobs = rec['n_jobs']
        n_machines = rec['n_machines']
        img_path = os.path.join(images_dir, f"{image_prefix}_{j:05d}_j{n_jobs}_m{n_machines}.png")
        if not os.path.exists(img_path):
            continue
        rows.append({
            'image_path': img_path,
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'operations': rec['operations'],
            'output': rec['output'],
            'optimal_makespan': rec.get('optimal_makespan', None),
        })
    return rows


def load_model(base_model, adapter_dir):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map='auto',
    )

    if adapter_dir and adapter_dir.lower() not in ('none', 'null', ''):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        processor = AutoProcessor.from_pretrained(adapter_dir)
    else:
        processor = AutoProcessor.from_pretrained(base_model)

    model.eval()
    return model, processor


def run_inference(model, processor, image_path, n_jobs, n_machines, max_new_tokens, instruction_template):
    from PIL import Image as PILImage
    image = PILImage.open(image_path).convert('RGB')

    if callable(instruction_template):
        instruction = instruction_template(int(n_jobs), int(n_machines))
    else:
        instruction = instruction_template.format(
            n_jobs=int(n_jobs),
            n_machines=int(n_machines)
        )

    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': instruction},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return processor.batch_decode(
        ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]


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
        return None, None, f"parse_error"

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
        return False, f"wrong_num_machines"

    all_jobs = set()
    for machine_schedule in schedule:
        if not isinstance(machine_schedule, list):
            return False, "machine_schedule_not_list"
        all_jobs.update(machine_schedule)

    expected_jobs = set(range(n))
    if all_jobs != expected_jobs:
        return False, f"job_coverage_error"

    return True, "ok"


def evaluate_jssp_solution(pred_text, instance_array, optimal_makespan=None):
    """Evaluate a JSSP solution."""
    schedule, makespan, parse_status = parse_solution(pred_text)

    if parse_status != "ok" or schedule is None:
        return False, float('inf'), float('inf')

    n_jobs = instance_array.shape[0]
    n_machines = instance_array.shape[1] // 2

    is_valid, val_status = validate_jssp_solution(schedule, n_jobs, n_machines)
    if not is_valid:
        return False, float('inf'), float('inf')

    # Compute actual makespan
    try:
        actual_makespan = get_makespan(instance_array, schedule)
        if actual_makespan == "infeasible":
            return False, float('inf'), float('inf')

        # Compute gap
        if optimal_makespan and optimal_makespan > 0:
            gap = (actual_makespan - optimal_makespan) / optimal_makespan
        else:
            gap = actual_makespan / 100.0

        return True, actual_makespan, gap
    except Exception:
        return False, float('inf'), float('inf')


def main():
    args = parse_args()
    rows = load_eval_rows(args.eval_json, args.images_dir, args.image_prefix, args.num_samples)
    if not rows:
        print('EVAL_SCORE:inf')
        return

    instruction_template = load_instruction_template(args.code_path)
    model, processor = load_model(args.base_model, args.adapter_dir)

    feasible_count = 0
    gaps = []

    for idx, row in enumerate(rows):
        pred_text = run_inference(
            model, processor, row['image_path'], row['n_jobs'],
            row['n_machines'], args.max_new_tokens, instruction_template,
        )

        # Convert operations to numpy array
        operations = row['operations']
        n_jobs = row['n_jobs']
        n_machines = row['n_machines']

        instance_array = np.zeros((n_jobs, 2 * n_machines))
        for job_idx in range(n_jobs):
            for op_idx in range(n_machines):
                machine_id, proc_time = operations[job_idx][op_idx]
                instance_array[job_idx, 2*op_idx] = machine_id
                instance_array[job_idx, 2*op_idx + 1] = proc_time

        is_feasible, makespan, gap = evaluate_jssp_solution(
            pred_text, instance_array, row.get('optimal_makespan')
        )

        if is_feasible:
            feasible_count += 1
            gaps.append(gap)
        else:
            gaps.append(args.infeasible_penalty)

        print(f'  [{idx+1}/{len(rows)}] feasible={is_feasible} makespan={makespan:.1f} gap={gap:.4f}')

    feasibility_rate = feasible_count / len(rows) if rows else 0.0

    # If all instances failed, return inf instead of penalty average
    if feasibility_rate == 0.0:
        mean_gap = float('inf')
    else:
        mean_gap = sum(gaps) / len(gaps) if gaps else float('inf')

    print(f'EVAL_DETAIL: feasibility_rate={feasibility_rate:.3f} mean_gap={mean_gap} n={len(rows)}')
    print(f'EVAL_SCORE:{mean_gap}')


if __name__ == '__main__':
    main()
