"""
generate_eval_images_jssp.py
=============================
Generate evaluation images for JSSP that match the training visualization format.
Reads test.json and creates corresponding JSSP visualization images.
Uses the SAME format as batch_generate_jssp_images.py
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm


# 定义颜色方案（每个作业使用不同颜色）- 与训练图像完全一致
COLORS = [
    '#5DCEA1',  # 青绿色 - Job 0
    '#FF9F80',  # 珊瑚橙色 - Job 1
    '#A3B8E6',  # 淡蓝色 - Job 2
    '#F0A8D8',  # 粉色 - Job 3
    '#D4E157',  # 黄绿色 - Job 4
    '#FFE082',  # 黄色 - Job 5
]


def parse_args():
    parser = argparse.ArgumentParser(description='Generate eval images for JSSP')
    parser.add_argument('--eval_json', type=str,
                        default='./data/sft/eval/test.json',
                        help='Path to eval JSON file')
    parser.add_argument('--output_dir', type=str,
                        default='./data/sft/eval/jsspgraph',
                        help='Output directory for images')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to generate (None = all)')
    return parser.parse_args()


def parse_job_data(input_str):
    """解析输入字符串，提取每个作业的机器和处理时间（与训练脚本一致）"""
    jobs = []
    job_parts = input_str.split('Job ')[1:]

    for job_part in job_parts:
        if 'machines and processing times for operations:' in job_part:
            start = job_part.find('[(')
            end = job_part.find(']', start) + 1
            operations_str = job_part[start:end]
            operations = eval(operations_str)
            jobs.append(operations)

    return jobs


def create_jssp_image(jobs_data, n_jobs, n_machines, output_path):
    """创建JSSP输入数据的可视化图像（与训练脚本完全一致）"""
    # 单元格大小（优化VLM处理）
    cell_width = 0.9
    cell_height = 0.75
    spacing_x = 0.15
    spacing_y = 0.12

    # 根据机器数量动态计算图像宽度
    fig_width = max(8, 0.8 + n_machines * (cell_width + spacing_x))
    fig_height = 1.6 + n_jobs * (cell_height + spacing_y)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('#F5F5F5')
    ax.set_facecolor('#F5F5F5')

    # 绘制标题
    title_y = n_jobs * (cell_height + spacing_y) + 1.0
    ax.text(n_machines * (cell_width + spacing_x) / 2, title_y + 0.3,
            f'JSSP Input — {n_jobs} Jobs × {n_machines} Machines',
            ha='center', va='top', fontsize=14, fontweight='normal')
    ax.text(n_machines * (cell_width + spacing_x) / 2, title_y,
            '(each cell: machine ID + processing time)',
            ha='center', va='top', fontsize=10, style='italic', color='#666')

    # 绘制操作列标签
    for op in range(n_machines):
        x = op * (cell_width + spacing_x) + cell_width / 2
        y = title_y - 0.6
        ax.text(x, y, f'Op{op+1}', ha='center', va='center',
                fontsize=11, fontweight='bold')

        if op < n_machines - 1:
            arrow_x = x + cell_width / 2 + 0.05
            ax.annotate('', xy=(arrow_x + 0.05, y), xytext=(arrow_x, y),
                       arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))

    # 绘制每个作业
    for job_idx, operations in enumerate(jobs_data):
        y = title_y - 1.2 - job_idx * (cell_height + spacing_y)

        ax.text(-0.3, y - cell_height / 2, f'J{job_idx}',
               ha='right', va='center', fontsize=11, fontweight='bold')

        for op_idx, (machine, time) in enumerate(operations):
            x = op_idx * (cell_width + spacing_x)

            color = COLORS[job_idx % len(COLORS)]
            rect = patches.Rectangle((x, y - cell_height), cell_width, cell_height,
                                    linewidth=1.5, edgecolor='white',
                                    facecolor=color, alpha=0.85)
            ax.add_patch(rect)

            ax.text(x + cell_width / 2, y - cell_height / 2 + 0.15,
                   f'M{machine}',
                   ha='center', va='center', fontsize=11, fontweight='bold')

            ax.text(x + cell_width / 2, y - cell_height / 2 - 0.15,
                   f't={time}',
                   ha='center', va='center', fontsize=9)

    ax.set_xlim(-0.8, n_machines * (cell_width + spacing_x))
    ax.set_ylim(-0.5, title_y + 0.6)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight',
                facecolor='#F5F5F5', edgecolor='none')
    plt.close()


def main():
    args = parse_args()

    # Load evaluation data
    print(f"Loading evaluation data from: {args.eval_json}")
    with open(args.eval_json) as f:
        data = json.load(f)

    if args.num_samples is not None:
        data = data[:args.num_samples]

    print(f"Total samples to generate: {len(data)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Generate images
    generated = 0
    skipped = 0

    for idx, rec in enumerate(tqdm(data, desc="Generating images")):
        n_jobs = int(rec['n'])
        n_machines = int(rec['m'])
        input_str = rec['input']

        # Parse job data using same method as training
        jobs_data = parse_job_data(input_str)

        # Generate filename matching training pattern
        filename = f"{idx:06d}_{n_jobs}x{n_machines}.png"
        output_path = output_dir / filename

        # Check if already exists
        if output_path.exists():
            skipped += 1
            continue

        try:
            create_jssp_image(jobs_data, n_jobs, n_machines, output_path)
            generated += 1

        except Exception as e:
            print(f"\n  [ERROR] Failed to generate {filename}: {e}")
            continue

    print("\n" + "="*60)
    print("IMAGE GENERATION SUMMARY")
    print("="*60)
    print(f"Total samples:    {len(data)}")
    print(f"Generated:        {generated}")
    print(f"Skipped (exists): {skipped}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
