#!/root/autodl-tmp/yao/llm_env/bin/python
"""
sft_eval.py for PFSP

Runs inference with a LoRA-fine-tuned VLM on PFSP eval instances.
"""

import argparse
import importlib.util
import json
import os
import re
import sys

import torch
import numpy as np

# Import PFSP utilities from VisionSolver-main
sys.path.insert(0, '/root/autodl-tmp/yao/VisionSolver-main-pfsp')
from Envs.PFSPEnv.baseline_algs import calculate_makespan

SYSTEM_PROMPT = (
    "You are an expert scheduling solver. "
    "You analyse visual Permutation Flow Shop Scheduling Problem (PFSP) instances and output "
    "optimal job sequences in the exact format requested."
)


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
    p.add_argument('--num_samples', type=int, default=30)
    p.add_argument('--max_new_tokens', type=int, default=1000)
    p.add_argument('--infeasible_penalty', type=float, default=1.0,
                    help='Penalty for infeasible solutions')
    p.add_argument('--code_path', type=str, required=True,
                    help='Path to candidate code file containing get_instruction function')
    return p.parse_args()


def parse_pfsp_input(input_text, n_machines):
    """Parse PFSP input text to extract processing times matrix."""
    processing_times = []
    machine_entries = input_text.split(';')

    for entry in machine_entries:
        entry = entry.strip()
        if not entry or 'Machine' not in entry:
            continue
        match = re.search(r'\[([\d,\s]+)\]', entry)
        if match:
            times_str = match.group(1)
            times = [int(x.strip()) for x in times_str.split(',') if x.strip()]
            processing_times.append(times)

    return processing_times


def load_eval_rows(eval_json, images_dir, image_prefix, num_samples):
    with open(eval_json) as f:
        data = json.load(f)
    rows = []
    for j, rec in enumerate(data[:num_samples]):
        n = int(rec['n']) if isinstance(rec['n'], str) else rec['n']
        m = int(rec['m']) if isinstance(rec['m'], str) else rec['m']
        img_path = os.path.join(images_dir, f"{image_prefix}_{j:05d}_j{n}_m{m}.png")
        if not os.path.exists(img_path):
            continue

        # Parse processing times from input text
        processing_times = parse_pfsp_input(rec['input'], m)

        rows.append({
            'image_path': img_path,
            'n': n,
            'm': m,
            'processing_times': processing_times,  # [machines × jobs]
            'output': rec['output'],
            'optimal_makespan': rec.get('optimal_makespan', None),
        })
    return rows


def load_model(base_model, adapter_dir):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import gc

    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Use 4-bit quantization to reduce memory usage
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map='auto',
        low_cpu_mem_usage=True,
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
        'role': 'system',
        'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]
    }, {
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

    # Extract schedule (job sequence)
    schedule_match = re.search(r'Schedule:\s*(\[[\d,\s]+\])', text)

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


def validate_pfsp_solution(schedule, n_jobs):
    """Validate if a PFSP schedule (job sequence) is well-formed."""
    if not isinstance(schedule, list):
        return False, "not_a_list"

    if len(schedule) != n_jobs:
        return False, f"wrong_length_{len(schedule)}_vs_{n_jobs}"

    # Check if it's a valid permutation
    if set(schedule) != set(range(n_jobs)):
        return False, "not_a_permutation"

    return True, "ok"


def evaluate_pfsp_solution(pred_text, processing_times, n_jobs, n_machines, optimal_makespan=None):
    """
    Evaluate a PFSP solution.
    processing_times: [machines × jobs]
    """
    schedule, makespan, parse_status = parse_solution(pred_text)

    if parse_status != "ok" or schedule is None:
        return False, float('inf'), float('inf')

    is_valid, val_status = validate_pfsp_solution(schedule, n_jobs)
    if not is_valid:
        return False, float('inf'), float('inf')

    # Compute actual makespan
    try:
        # Convert processing_times to numpy array [machines × jobs]
        pt_array = np.array(processing_times, dtype=np.int32)

        # Calculate makespan for the given sequence
        # Convert 0-based schedule to 1-based for calculate_makespan
        # Transpose to [jobs × machines] as expected by calculate_makespan
        schedule_1based = [x + 1 for x in schedule]
        actual_makespan = calculate_makespan(schedule_1based, pt_array.T)

        if actual_makespan == "infeasible" or actual_makespan < 0:
            return False, float('inf'), float('inf')

        # Compute gap
        if optimal_makespan and optimal_makespan > 0:
            gap = (actual_makespan - optimal_makespan) / optimal_makespan
        else:
            gap = actual_makespan / 100.0

        return True, actual_makespan, gap
    except Exception as e:
        print(f'    EXCEPTION in evaluate_pfsp_solution: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
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
            model, processor, row['image_path'], row['n'],
            row['m'], args.max_new_tokens, instruction_template,
        )

        is_feasible, makespan, gap = evaluate_pfsp_solution(
            pred_text, row['processing_times'], row['n'], row['m'],
            row.get('optimal_makespan')
        )

        if is_feasible:
            feasible_count += 1
            gaps.append(gap)
        else:
            gaps.append(args.infeasible_penalty)

        print(f'  [{idx+1}/{len(rows)}] pred="{pred_text[:80]}" feasible={is_feasible} makespan={makespan:.1f} gap={gap:.4f}')
        if not is_feasible and idx < 3:  # Print first 3 failed predictions for debugging
            print(f'    VLM_OUTPUT: {pred_text[:200]}')

    feasibility_rate = feasible_count / len(rows) if rows else 0.0
    mean_gap = sum(gaps) / len(gaps) if gaps else float('inf')

    print(f'EVAL_DETAIL: feasibility_rate={feasibility_rate:.3f} mean_gap={mean_gap:.4f} n={len(rows)}')
    print(f'EVAL_SCORE:{mean_gap}')


if __name__ == '__main__':
    main()
