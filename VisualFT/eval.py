"""
eval_vision_grid5000.py

Evaluates the LoRA model trained by main_train_vision_grid5000.py on 50
held-out instances from test_under50nodes.json.

Steps:
  1. Render 50 eval images using the same smallfig draw style as training.
  2. Load the trained LoRA checkpoint.
  3. Run inference with the same build_instruction() used during training.
  4. Score with compute_metric_cop (feasibility_rate, mean_gap).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, '/root/autodl-tmp/yao/LLMCoSolver-main')
from utils import compute_metric_cop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_200k_smallfig import draw, parse_nodes as smallfig_parse_nodes

import re


def parse_nodes(input_str):
    nodes = {}
    for m in re.finditer(
        r'Node (\d+), coordinates: \[(\d+), (\d+)\], demand: (\d+)', input_str
    ):
        nid = int(m.group(1))
        nodes[nid] = {'x': int(m.group(2)), 'y': int(m.group(3)), 'demand': int(m.group(4))}
    return nodes


def build_instruction(vehicle_capacity, num_nodes=None) -> str:
    cap = int(float(vehicle_capacity))
    n_customers = (int(num_nodes) - 1) if num_nodes is not None else None
    node_hint = f"There are {n_customers} customers (nodes 1 to {n_customers}). " if n_customers is not None else ""
    return (
        f"The image shows a Capacitated Vehicle Routing Problem (CVRP) instance "
        f"on a 25x25 grid. "
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


def render_eval_images(eval_json, out_dir, num_samples):
    os.makedirs(out_dir, exist_ok=True)
    with open(eval_json) as f:
        data = json.load(f)
    data = data[:num_samples]
    skipped = 0
    for idx, rec in enumerate(data):
        num_nodes = int(rec['num_nodes'])
        capacity = float(rec['vehicle_capacity'])
        out_path = os.path.join(out_dir, f'eval_{idx:06d}_n{num_nodes}.png')
        if os.path.exists(out_path):
            skipped += 1
            continue
        nodes = smallfig_parse_nodes(rec['input'])
        draw(nodes, num_nodes, capacity, out_path)
    n_done = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    print(f"Eval images: {n_done}/{num_samples} ({skipped} skipped/existing)")
    return n_done


def load_eval_rows(eval_json, images_dir, num_samples):
    with open(eval_json) as f:
        data = json.load(f)
    rows = []
    for idx, rec in enumerate(data[:num_samples]):
        num_nodes = int(rec['num_nodes'])
        img_path = os.path.join(images_dir, f'eval_{idx:06d}_n{num_nodes}.png')
        if not os.path.exists(img_path):
            continue
        rows.append({
            'image_path': img_path,
            'vehicle_capacity': str(rec['vehicle_capacity']),
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


def run_inference(model, processor, image_path, vehicle_capacity, num_nodes, max_new_tokens):
    from PIL import Image as PILImage
    image = PILImage.open(image_path).convert('RGB')
    instruction = build_instruction(vehicle_capacity, int(num_nodes))
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': instruction},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(
        ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', type=str,
                   default='/root/autodl-tmp/yao/models1_cache/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master')
    p.add_argument('--adapter_dir', type=str,
                   default='/root/autodl-tmp/yao/VisionSolver-main/output_vision_grid_alpha64_r64_cvrp_seq20000_b4_ep1/checkpoint-312')
    p.add_argument('--eval_json', type=str,
                   default='/root/autodl-tmp/yao/VDEvo/problems/vision_gen/test_under50nodes.json')
    p.add_argument('--eval_images_dir', type=str,
                   default='/root/autodl-tmp/yao/VisionSolver-main/data/sft/eval/images_smallfig_under50nodes')
    p.add_argument('--num_samples', type=int, default=50)
    p.add_argument('--max_new_tokens', type=int, default=3000)
    p.add_argument('--infeasible_penalty', type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Rendering eval images to {args.eval_images_dir} ...")
    n = render_eval_images(args.eval_json, args.eval_images_dir, args.num_samples)
    if n == 0:
        print("EVAL_SCORE:inf")
        return

    print("Loading eval rows ...")
    rows = load_eval_rows(args.eval_json, args.eval_images_dir, args.num_samples)
    print(f"  {len(rows)} rows loaded.")

    print("Loading model ...")
    model, processor = load_model(args.base_model, args.adapter_dir)

    predictions, labels, instances = [], [], []
    for i, row in enumerate(rows):
        pred_text = run_inference(
            model, processor, row['image_path'],
            row['vehicle_capacity'], row['num_nodes'], args.max_new_tokens,
        )
        predictions.append('### Response:\n' + pred_text)
        labels.append(row['output'])
        nodes = row['nodes']
        locs = [(nodes[k]['x'], nodes[k]['y']) for k in sorted(nodes.keys())]
        demands = [nodes[k]['demand'] for k in sorted(nodes.keys())]
        instances.append([locs, demands, float(row['vehicle_capacity'])])
        if (i + 1) % 10 == 0:
            print(f"  inference {i + 1}/{len(rows)}", flush=True)

    feasibility_rate, mean_gap, std_gap = compute_metric_cop(
        predictions, labels, instances, problem='cvrp'
    )
    infeasibility_penalty = (1.0 - feasibility_rate) * args.infeasible_penalty
    fitness = mean_gap + infeasibility_penalty

    print(f"EVAL_DETAIL: feasibility_rate={feasibility_rate:.3f} mean_gap={mean_gap:.4f} std_gap={std_gap:.4f} n={len(rows)}")
    print(f"EVAL_SCORE:{fitness}")


if __name__ == '__main__':
    main()
