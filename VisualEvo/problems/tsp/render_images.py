#!/root/autodl-tmp/yao/llm_env/bin/python
"""
render_images.py for TSP

Renders a training/eval image dataset for the TSP VDEvo problem.

Given:
  - a candidate drawing code file (gpt.py) containing draw(nodes, num_nodes, out_path)
  - a source json file of TSP instances (fields: num_nodes, input, [output])

Produces:
  - PNG images in out_dir, named {prefix}_{idx:05d}_n{num_nodes}.png

Usage:
  python render_images.py --code_path gpt.py --json_path data.json --out_dir imgs \
      --count 2000 --workers 12 --prefix train
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from multiprocessing import Pool


def parse_nodes(input_str):
    """Parse TSP node coordinates from input string."""
    nodes = {}
    for m in re.finditer(
        r'Node (\d+), coordinates: \[(\d+), (\d+)\]', input_str
    ):
        nid = int(m.group(1))
        nodes[nid] = {
            'x': int(m.group(2)),
            'y': int(m.group(3)),
        }
    return nodes


def load_draw_fn(code_path):
    spec = importlib.util.spec_from_file_location("candidate_gpt", code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    draw_fn = None
    for name in ("draw_v2", "draw_v1", "draw"):
        if hasattr(module, name):
            draw_fn = getattr(module, name)
            break
    if draw_fn is None:
        raise AttributeError(f"No draw_v2/draw_v1/draw function found in {code_path}")

    # Verify get_instruction function exists (check prompt.py first, then gpt.py)
    prompt_path = code_path.replace('gpt.py', 'prompt.py')
    if os.path.exists(prompt_path):
        instr_spec = importlib.util.spec_from_file_location("candidate_prompt", prompt_path)
        instr_module = importlib.util.module_from_spec(instr_spec)
        instr_spec.loader.exec_module(instr_module)
    else:
        instr_module = module

    instr_fn = None
    for name in ("get_instruction_v2", "get_instruction_v1", "get_instruction"):
        if hasattr(instr_module, name):
            instr_fn = getattr(instr_module, name)
            break
    if instr_fn is None:
        raise AttributeError(f"No get_instruction function found in {code_path} or prompt.py")

    return draw_fn


_WORKER_STATE = {}


def _init_worker(code_path):
    _WORKER_STATE["draw_fn"] = load_draw_fn(code_path)


def _worker(args_tuple):
    idx, rec, out_dir, prefix = args_tuple
    num_nodes = int(rec['num_nodes'])
    out_path = os.path.join(out_dir, f'{prefix}_{idx:05d}_n{num_nodes}.png')

    if os.path.exists(out_path):
        return idx, True, 'skip'

    nodes = parse_nodes(rec['input'])
    if not nodes:
        return idx, False, 'parse error'

    try:
        _WORKER_STATE["draw_fn"](nodes, num_nodes, out_path)
        return idx, True, 'ok'
    except Exception as e:
        return idx, False, str(e)


def render(code_path, json_path, out_dir, count, workers, prefix):
    os.makedirs(out_dir, exist_ok=True)
    with open(json_path) as f:
        data = json.load(f)
    data = data[:count]

    tasks = [(i, rec, out_dir, prefix) for i, rec in enumerate(data)]

    errors = 0
    if workers <= 1:
        _init_worker(code_path)
        for t in tasks:
            _, ok, msg = _worker(t)
            if not ok and msg != 'skip':
                errors += 1
    else:
        with Pool(processes=workers, initializer=_init_worker, initargs=(code_path,)) as pool:
            for _, ok, msg in pool.imap_unordered(_worker, tasks, chunksize=4):
                if not ok and msg != 'skip':
                    errors += 1

    n_done = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    return n_done, errors, len(tasks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code_path', required=True)
    p.add_argument('--json_path', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--count', type=int, default=5000)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--prefix', type=str, default='train')
    args = p.parse_args()

    n_done, errors, total = render(
        args.code_path, args.json_path, args.out_dir, args.count, args.workers, args.prefix
    )
    print(f'RENDER_DONE:{n_done}/{total} errors={errors}')
    if n_done == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
