#!/usr/bin/env python3
"""
filter_jssp_datasets.py

Filter existing JSSP datasets to create smaller subsets with n*m <= 55.
- Training: 5000 instances from training_jssp-001.json
- Test: 50 instances from test.json (disjoint from training)
"""

import json
import re
from tqdm import tqdm

MAX_PRODUCT = 60


def parse_operations(input_field: str, n: int, m: int):
    """Parse the real per-job (machine, time) operation pairs from the input string."""
    job_blocks = re.findall(r'machines and processing times for operations:\s*\[(.*?)\]', input_field)
    if len(job_blocks) != n:
        return None

    operations = []
    for jb in job_blocks:
        pairs = re.findall(r'\((\d+),\s*(\d+)\)', jb)
        if len(pairs) != m:
            return None
        operations.append([[int(machine), int(time)] for machine, time in pairs])
    return operations


def filter_from_file(source_file, target_count):
    """Filter records from a source file."""
    print(f"Loading {source_file}...")
    with open(source_file) as f:
        data = json.load(f)
    print(f"Total records in source: {len(data)}")

    filtered = []
    skipped_parse = 0

    for rec in tqdm(data, desc=f"Filtering {source_file.split('/')[-1]}"):
        if len(filtered) >= target_count:
            break

        n, m = int(rec['n']), int(rec['m'])
        if n * m > MAX_PRODUCT:
            continue

        operations = parse_operations(rec['input'], n, m)
        if operations is None:
            skipped_parse += 1
            continue

        entry = {
            'n_jobs': n,
            'n_machines': m,
            'operations': operations,
            'output': rec['output'],
        }
        filtered.append(entry)

    if skipped_parse:
        print(f"Skipped {skipped_parse} records (failed to parse operations).")

    return filtered


def main():
    train_source = '/root/autodl-tmp/yao/VisionSolver-main-jssp/data/sft/train/training_jssp-001.json'
    test_source = '/root/autodl-tmp/yao/VisionSolver-main-jssp/data/sft/eval/test.json'

    train_output = '/root/autodl-tmp/yao/Hercules/problems/vision_jssp/train_jssp_5000_under60.json'
    test_output = '/root/autodl-tmp/yao/Hercules/problems/vision_jssp/test_jssp_50_under60.json'

    train_count_target = 5000
    test_count_target = 50

    # Filter training set
    train_filtered = filter_from_file(train_source, train_count_target)
    print(f"Train: {len(train_filtered)} instances with n*m <= {MAX_PRODUCT}")

    # Filter test set
    test_filtered = filter_from_file(test_source, test_count_target)
    print(f"Test:  {len(test_filtered)} instances with n*m <= {MAX_PRODUCT}")

    # Save
    with open(train_output, 'w') as f:
        json.dump(train_filtered, f, indent=2)
    with open(test_output, 'w') as f:
        json.dump(test_filtered, f, indent=2)

    print(f"\nSaved to {train_output}")
    print(f"Saved to {test_output}")


if __name__ == '__main__':
    main()
