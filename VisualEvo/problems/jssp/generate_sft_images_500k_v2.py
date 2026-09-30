#!/usr/bin/env python3
"""
生成500,000张SFT训练图像
使用 09120001_ID_130636_1.00 的设计风格
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing as mp
import sys


# 定义颜色方案（与参考代码一致）
COLORS = [
    '#D3E3FC',  # 浅蓝色
    '#FFBFC0',  # 浅粉色
    '#D1F1DE',  # 浅绿色
    '#FFE356',  # 黄色
    '#F7B8DB',  # 浅紫粉色
    '#BCD4E6',  # 浅蓝灰色
    '#F696A2',  # 粉红色
    '#C5E1A5',  # 浅绿色
    '#FFB3B3',  # 浅橙粉色
    '#B39DDB',  # 浅紫色
    '#FFCC80',  # 浅橙色
    '#80DEEA',  # 青色
]


def parse_job_data(input_str):
    """解析输入字符串，提取每个作业的机器和处理时间"""
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
    """创建JSSP输入数据的可视化图像（使用参考代码的设计风格）"""

    # 缩小到60%以匹配CVRP的图像大小
    scale = 0.6
    cell_width = 1.0 * scale
    cell_height = 0.9 * scale
    spacing_x = 0.25 * scale
    spacing_y = 0.15 * scale

    fig_width = max(9, 0.9 + n_machines * (cell_width + spacing_x))
    fig_height = 1.5 + n_jobs * (cell_height + spacing_y)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    title_y = n_jobs * (cell_height + spacing_y) + 1.2 * scale
    ax.text(n_machines * (cell_width + spacing_x) / 2, title_y + 0.3 * scale,
            f'Job Shop Scheduling Problem Visualization: {n_jobs} Jobs × {n_machines} Machines',
            ha='center', va='top', fontsize=16, fontweight='bold', color='#4A4A4A')
    ax.text(n_machines * (cell_width + spacing_x) / 2, title_y,
            'Machine ID (M) includes time (t), coded color={ Job Index }',
            ha='center', va='top', fontsize=10, style='italic', color='#777777')

    for op in range(n_machines):
        x = op * (cell_width + spacing_x) + cell_width / 2
        y = title_y - 0.7 * scale
        ax.text(x, y, f'Op{op+1}', ha='center', va='center',
                fontsize=12, fontweight='bold', color='#202020')

        if op < n_machines - 1:
            arrow_x = x + cell_width / 2 + 0.05 * scale
            ax.annotate('', xy=(arrow_x + 0.05 * scale, y), xytext=(arrow_x, y),
                       arrowprops=dict(arrowstyle='->', color='#999999', lw=1.8 * scale))

    for job_idx in range(n_jobs):
        job_ops = jobs_data[job_idx]
        y = title_y - 1.5 * scale - job_idx * (cell_height + spacing_y)

        ax.text(-0.6 * scale, y - cell_height / 2, f'Job {job_idx + 1}',
               ha='right', va='center', fontsize=12, fontweight='bold', color='#313131')

        for op_idx, (machine, time) in enumerate(job_ops):
            x = op_idx * (cell_width + spacing_x)

            color = COLORS[job_idx % len(COLORS)]
            rect = patches.Rectangle((x, y - cell_height), cell_width, cell_height,
                                    linewidth=2 * scale, edgecolor='black',
                                    facecolor=color, alpha=0.95)
            ax.add_patch(rect)

            ax.text(x + cell_width / 2, y - cell_height / 2 + 0.18 * scale,
                   f'M{machine}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')

            ax.text(x + cell_width / 2, y - cell_height / 2 - 0.13 * scale,
                   f't={time}', ha='center', va='center', fontsize=10, color='black')

    ax.set_xlim(-0.8 * scale, n_machines * (cell_width + spacing_x))
    ax.set_ylim(-0.6 * scale, title_y + 0.4 * scale)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight',
                facecolor='#FAFAFA', edgecolor='none')
    plt.close()


def process_single_item(args):
    """处理单个数据项（用于多进程）"""
    idx, item, output_dir = args

    try:
        n_jobs = int(item['n'])
        n_machines = int(item['m'])
        input_str = item['input']

        jobs_data = parse_job_data(input_str)

        output_path = output_dir / f"{idx:06d}_{n_jobs}x{n_machines}.png"
        create_jssp_image(jobs_data, n_jobs, n_machines, output_path)

        return True, idx
    except Exception as e:
        return False, (idx, str(e))


def main():
    parser = argparse.ArgumentParser(description='生成500K SFT训练图像（v2设计风格）')
    parser.add_argument('--json', type=str,
                       default='data/sft/train/training_jssp-001.json',
                       help='JSON数据文件路径')
    parser.add_argument('--output-dir', type=str,
                       default='data/sft/train/',
                       help='输出目录')
    parser.add_argument('--start', type=int, default=0,
                       help='起始索引')
    parser.add_argument('--end', type=int, default=None,
                       help='结束索引（不包含），默认处理所有数据')
    parser.add_argument('--workers', type=int, default=8,
                       help='并行进程数')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='每批处理的数量')

    args = parser.parse_args()

    # 读取JSON数据
    print(f"读取数据: {args.json}")
    with open(args.json, 'r') as f:
        data = json.load(f)

    # 确定处理范围
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(data)
    end_idx = min(end_idx, len(data))

    print(f"数据总数: {len(data)}")
    print(f"处理范围: {start_idx} 到 {end_idx}")
    print(f"使用 {args.workers} 个进程")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 分批处理
    total_samples = end_idx - start_idx
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size

    print(f"\n开始生成图像，共 {num_batches} 批...")
    print(f"输出目录: {output_dir.absolute()}")

    total_success = 0
    total_failed = []

    for batch_idx in range(num_batches):
        batch_start = start_idx + batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, end_idx)

        print(f"\n批次 {batch_idx + 1}/{num_batches}: 索引 {batch_start} - {batch_end}")

        # 准备任务列表
        tasks = [(idx, data[idx], output_dir) for idx in range(batch_start, batch_end)]

        # 使用多进程批量生成
        with mp.Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_item, tasks),
                total=len(tasks),
                desc=f"批次 {batch_idx + 1}"
            ))

        # 统计结果
        success_count = sum(1 for success, _ in results if success)
        failed = [result[1] for result in results if not result[0]]

        total_success += success_count
        total_failed.extend(failed)

        print(f"批次 {batch_idx + 1} 完成: 成功 {success_count}/{len(tasks)}")

    print(f"\n全部生成完成!")
    print(f"总成功: {total_success}/{total_samples}")

    if total_failed:
        print(f"总失败: {len(total_failed)}")
        print("失败的索引（前10个）:")
        for idx, err in total_failed[:10]:
            print(f"  - 索引 {idx}: {err}")


if __name__ == '__main__':
    main()
