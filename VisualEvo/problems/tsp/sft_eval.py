#!/root/autodl-tmp/yao/llm_env/bin/python
"""
sft_eval.py for TSP

Runs inference with a LoRA-fine-tuned VLM on a fixed, held-out set of TSP
eval instances (with known ground-truth objective) and computes a fitness
score (mean optimality gap) for the TSP VDEvo problem.

Uses the same compute_metric_cop evaluation function as VisionSolver-main.
"""

import argparse
import importlib.util
import json
import os
import re
import sys

import torch

# Import compute_metric_cop from LLMCoSolver-main
sys.path.insert(0, '/root/autodl-tmp/yao/LLMCoSolver-main')
from utils import compute_metric_cop


def load_instruction_template(code_path):
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


def parse_nodes(input_str):
    nodes = {}
    for m in re.finditer(
        r'City (\d+), coordinates: \[(\d+), (\d+)\]', input_str
    ):
        nid = int(m.group(1))
        nodes[nid] = {
            'x': int(m.group(2)),
            'y': int(m.group(3)),
        }
    return nodes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', type=str,
                    default='/root/autodl-tmp/yao/models1_cache/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master')
    p.add_argument('--adapter_dir', type=str, required=True)
    p.add_argument('--eval_json', type=str, required=True)
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--image_prefix', type=str, default='eval')
    p.add_argument('--num_samples', type=int, default=20)
    p.add_argument('--max_new_tokens', type=int, default=3000)
    p.add_argument('--code_path', type=str, required=True,
                    help='Path to candidate code file containing get_instruction function')
    return p.parse_args()


def load_eval_rows(eval_json, images_dir, image_prefix, num_samples):
    with open(eval_json) as f:
        data = json.load(f)
    rows = []
    for j, rec in enumerate(data[:num_samples]):
        img_path = os.path.join(images_dir, f"{image_prefix}_{j:05d}_n{rec['num_nodes']}.png")
        if not os.path.exists(img_path):
            continue
        rows.append({
            'image_path': img_path,
            'num_nodes': rec['num_nodes'],
            'output': rec['output'],
            'nodes': parse_nodes(rec['input']),
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


def run_inference(model, processor, image_path, num_nodes, max_new_tokens, instruction_template):
    from PIL import Image as PILImage
    image = PILImage.open(image_path).convert('RGB')

    # Handle both string template and function
    if callable(instruction_template):
        instruction = instruction_template(int(num_nodes))
    else:
        instruction = instruction_template.format(num_nodes=int(num_nodes))

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


def main():
    args = parse_args()
    rows = load_eval_rows(args.eval_json, args.images_dir, args.image_prefix, args.num_samples)
    if not rows:
        print('EVAL_SCORE:inf')
        return

    instruction_template = load_instruction_template(args.code_path)
    model, processor = load_model(args.base_model, args.adapter_dir)

    predictions = []
    labels = []
    instances = []

    for row in rows:
        pred_text = run_inference(
            model, processor, row['image_path'], row['num_nodes'],
            args.max_new_tokens, instruction_template,
        )
        # Format prediction as VisionSolver expects: "### Response:\n" + text
        predictions.append('### Response:\n' + pred_text)
        # Labels are the ground truth output strings
        labels.append(row['output'])
        # Instances are in format [locs] for TSP
        nodes = row['nodes']
        locs = [(nodes[i]['x'], nodes[i]['y']) for i in sorted(nodes.keys())]
        instances.append([locs])

    # Use VisionSolver's compute_metric_cop function
    feasibility_rate, mean_gap, std_gap = compute_metric_cop(
        predictions, labels, instances, problem='tsp'
    )

    print(f'EVAL_DETAIL: feasibility_rate={feasibility_rate:.3f} mean_gap={mean_gap:.4f} std_gap={std_gap:.4f} n={len(rows)}')

    # For TSP, fitness is just mean_gap (all tours are feasible)
    fitness = mean_gap

    print(f'EVAL_SCORE:{fitness}')


if __name__ == '__main__':
    main()
