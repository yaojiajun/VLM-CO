"""
Generate 200,000 images using the smallfig style.
"""

import json
import os
import re
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

COORD_MAX = 1000


def parse_nodes(input_str):
    nodes = {}
    for m in re.finditer(
        r'Node (\d+), coordinates: \[(\d+), (\d+)\], demand: (\d+)', input_str
    ):
        nid = int(m.group(1))
        nodes[nid] = {
            'x': int(m.group(2)),
            'y': int(m.group(3)),
            'demand': int(m.group(4)),
        }
    return nodes


def resolve_conflicts(raw, gs):
    placed, positions = {}, {}
    for nid, (gx, gy) in raw:
        cell = (gx, gy)
        if cell not in placed:
            placed[cell] = nid
            positions[nid] = cell
        else:
            found = False
            for radius in range(1, gs):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if abs(dr) != radius and abs(dc) != radius:
                            continue
                        nc, nr = gx + dc, gy + dr
                        if 0 <= nc < gs and 0 <= nr < gs and (nc, nr) not in placed:
                            placed[(nc, nr)] = nid
                            positions[nid] = (nc, nr)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                positions[nid] = (gx, gy)
    return positions


def draw(nodes, num_nodes, capacity, out_path):
    nodes_info = [(nid, v['x'], v['y'], v['demand']) for nid, v in nodes.items()]
    grid_size  = 25

    def to_grid(x, y):
        gx = round(x / COORD_MAX * (grid_size - 1))
        gy = round(y / COORD_MAX * (grid_size - 1))
        return gx, gy

    raw_grid   = [(nid, to_grid(x, y)) for nid, x, y, _ in nodes_info]
    positions  = resolve_conflicts(raw_grid, grid_size)
    demand_map = {nid: d for nid, _, _, d in nodes_info}
    max_demand = max((d for d in demand_map.values() if d > 0), default=1)
    min_demand = min((d for d in demand_map.values() if d > 0), default=1)

    fig_in    = max(6, min(18, grid_size * 22 / 100))
    base_ms   = max(3, min(11, 140 / grid_size))
    font_size = max(3.5, min(8, 120 / grid_size))

    fig, ax = plt.subplots(figsize=(fig_in, fig_in), facecolor='white')
    ax.set_facecolor('white')

    for k_line in range(grid_size + 1):
        ax.axhline(k_line - 0.5, color='#b0b0b0', lw=0.6, zorder=0)
        ax.axvline(k_line - 0.5, color='#b0b0b0', lw=0.6, zorder=0)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')

    pts_per_data = fig_in * 72 / grid_size

    for nid, (col, row) in positions.items():
        d = demand_map[nid]
        if nid == 0:
            ax.plot(col, row, '*', color='#2ca02c', markersize=base_ms * 2.5,
                    markeredgecolor='darkgreen', markeredgewidth=0.8, zorder=4)
            depot_offset = (base_ms * 2.5 / 2.0) / pts_per_data + 0.05
            ax.text(col, row + depot_offset, 'Depot',
                    ha='center', va='bottom',
                    fontsize=font_size, color='#2ca02c',
                    fontweight='semibold', zorder=5)
        else:
            t = (d - min_demand) / max(max_demand - min_demand, 1)
            size = base_ms * (0.4 + 1.2 * t)
            ax.plot(col, row, 'o', color='#6baed6', markersize=size,
                    markeredgecolor='#2171b5', markeredgewidth=0.6, zorder=3)
            offset = (size / 2.0) / pts_per_data + 0.05
            ax.text(col, row + offset, f'C{nid}[{d}]',
                    ha='center', va='bottom',
                    fontsize=max(3.0, font_size * 0.78), color='black',
                    fontweight='bold', zorder=5)

    ax.set_title(
        f'Blue●=customer (C{{id}}[demand] above), ★=depot\n'
        f'Nodes: {num_nodes}  |  Cap: {int(capacity)}  |  Grid: {grid_size}×{grid_size}',
        fontsize=max(7, font_size + 1), pad=6,
    )

    tick_pos = list(range(0, 25, 2))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_pos)
    ax.tick_params(labelsize=max(5, font_size - 1))
    ax.set_xlabel('x', fontsize=max(6, font_size))
    ax.set_ylabel('y', fontsize=max(6, font_size))

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)


def worker(args_tuple):
    idx, rec, out_dir = args_tuple
    num_nodes = int(rec['num_nodes'])
    capacity = float(rec['vehicle_capacity'])
    out_path = os.path.join(out_dir, f'train_{idx:06d}_n{num_nodes}.png')

    if os.path.exists(out_path):
        return idx, True, 'skip'

    nodes = parse_nodes(rec['input'])
    if not nodes:
        return idx, False, 'parse error'

    try:
        draw(nodes, num_nodes, capacity, out_path)
        return idx, True, 'ok'
    except Exception as e:
        return idx, False, str(e)


def main():
    json_path = './data/sft/train_cvrp-001.json'
    out_dir = './data/sft/train/images_500k'
    count = 10000
    workers = 12

    os.makedirs(out_dir, exist_ok=True)

    print(f'读取数据集...')
    with open(json_path) as f:
        data = json.load(f)

    data = data[:count]
    print(f'生成 {len(data)} 张图像 -> {out_dir}  (workers={workers})')

    tasks = [(i, rec, out_dir) for i, rec in enumerate(data)]

    errors = 0
    with Pool(processes=workers) as pool:
        for _, ok, msg in tqdm(
            pool.imap_unordered(worker, tasks, chunksize=8),
            total=len(tasks),
            desc='生成图像'
        ):
            if not ok and msg != 'skip':
                errors += 1
                if msg != 'parse error':
                    print(f'  error: {msg}')

    n_done = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    print(f'✅ 完成: {n_done} 张图像 ({errors} errors)')


if __name__ == '__main__':
    main()
