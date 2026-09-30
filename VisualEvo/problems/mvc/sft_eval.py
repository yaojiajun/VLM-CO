#!/root/autodl-tmp/yao/llm_env/bin/python
"""
sft_eval.py

Runs inference with a LoRA-fine-tuned VLM on a fixed, held-out set of MVC
eval instances and computes a fitness score (exact match accuracy and objective accuracy)
for the vision_solver_main_mvc Hercules problem.
"""

import argparse
import importlib.util
import json
import os
import re
import sys

import torch

# Import compute metrics from local utils
from utils import compute_metric_mvc


def load_instruction_template(code_path):
    # For MIS vision problem, load from prompt.py
    prompt_path = code_path.replace('gpt.py', 'prompt.py')
    if os.path.exists(prompt_path):
        code_path = prompt_path

    spec = importlib.util.spec_from_file_location("candidate_prompt", code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("get_instruction_v2", "get_instruction_v1", "get_instruction"):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"No get_instruction function found in {code_path}")


SYSTEM_PROMPT = (
    "You are an expert combinatorial optimization solver. "
    "You analyze visual graph problem instances and output the minimum vertex cover "
    "in the exact format requested."
)


def parse_edges(input_str):
    """Parse edges from input string"""
    edges = []
    edges_match = re.search(r'Edges:\s*\[(.*?)\]', input_str, re.DOTALL)
    if edges_match:
        edges_str = edges_match.group(1)
        for edge in edges_str.split('),('):
            edge = edge.strip('()')
            if ',' in edge:
                parts = edge.split(',')
                try:
                    a, b = int(parts[0].strip()), int(parts[1].strip())
                    edges.append((a, b))
                except:
                    continue
    return edges


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', type=str,
                    default='/root/autodl-tmp/yao/models1_cache/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master')
    p.add_argument('--adapter_dir', type=str, required=True)
    p.add_argument('--eval_json', type=str, required=True)
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--image_prefix', type=str, default='eval')
    p.add_argument('--num_samples', type=int, default=20)
    p.add_argument('--max_new_tokens', type=int, default=512)
    p.add_argument('--code_path', type=str, required=True,
                    help='Path to candidate code file containing get_instruction function')
    return p.parse_args()


def load_eval_rows(eval_json, images_dir, image_prefix, num_samples):
    with open(eval_json) as f:
        data = json.load(f)
    rows = []
    for j, rec in enumerate(data[:num_samples]):
        num_nodes = int(rec['num_nodes'])
        img_path = os.path.join(images_dir, f"{image_prefix}_{j:06d}_{num_nodes}.png")
        if not os.path.exists(img_path):
            continue
        edges = parse_edges(rec['input'])
        rows.append({
            'image_path': img_path,
            'num_nodes': num_nodes,
            'output': rec['output'],
            'edges': edges,
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

    messages = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': SYSTEM_PROMPT}],
        },
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': instruction},
            ],
        },
    ]
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
        print('EVAL_SCORE:0.0')
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
        # Format prediction as VisionSolver expects
        predictions.append(pred_text)
        # Labels are the ground truth output strings
        labels.append(row['output'])
        # Instances are [num_nodes, edges]
        instances.append([row['num_nodes'], row['edges']])

    # Compute detailed metrics for better fitness evaluation
    from utils import parse_mvc_output, is_vertex_cover

    exact_match = 0
    total_gap = 0.0  # Sum of (predicted_size - optimal_size) / optimal_size
    infeasible_count = 0
    total = len(predictions)

    for pred_text, label_text, instance in zip(predictions, labels, instances):
        pred_set, pred_obj = parse_mvc_output(pred_text)
        true_set, true_obj = parse_mvc_output(label_text)
        num_nodes, edges = instance

        # Check exact match
        if sorted(pred_set) == sorted(true_set):
            exact_match += 1

        # Compute normalized gap: (predicted - optimal) / optimal
        # For MVC: smaller is better, so gap > 0 means predicted > optimal (bad)
        if true_obj > 0:
            gap = (pred_obj - true_obj) / true_obj
            total_gap += max(0, gap)  # Only count when predicted > optimal

        # Check feasibility: predicted set should be a valid vertex cover
        # (every edge has at least one endpoint in pred_set)
        is_feasible = is_vertex_cover(pred_set, edges)

        if not is_feasible or pred_obj < true_obj:
            infeasible_count += 1

    exact_acc = exact_match / total if total > 0 else 0.0
    mean_gap = total_gap / total if total > 0 else 0.0
    feasibility_rate = 1.0 - (infeasible_count / total) if total > 0 else 0.0

    print(f'EVAL_DETAIL: exact_acc={exact_acc:.3f} mean_gap={mean_gap:.4f} feasibility={feasibility_rate:.3f} n={total}')

    # Fitness similar to vision_gen: mean_gap + infeasibility_penalty
    # Lower is better for Hercules (minimization)
    infeasibility_penalty = (1.0 - feasibility_rate) * 1.0
    fitness = mean_gap + infeasibility_penalty

    print(f'EVAL_SCORE:{fitness}')


if __name__ == '__main__':
    main()
