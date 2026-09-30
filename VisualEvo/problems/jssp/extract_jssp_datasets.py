#!/usr/bin/env python3
"""
extract_jssp_datasets.py

Extract JSSP datasets from existing data and convert to proper format.
- Training: 5000 instances with n*m < 30
- Test: 50 instances with n*m < 30
"""

import json
import re
from tqdm import tqdm


def parse_operations_from_input(input_str, n, m):
    """Parse operations from input string."""
    operations = []

    # Pattern: "Job X, machines and processing times for operations: [(machine, time), ...]"
    job_pattern = r'Job (\d+), machines and processing times for operations: \[(.*?)\](?:,|;|\.|$)'

    matches = re.finditer(job_pattern, input_str)

    for match in matches:
        job_id = int(match.group(1))
        ops_str = match.group(2)

        # Parse tuples: (machine, time)
        tuple_pattern = r'\((\d+),\s*(\d+)\)'
        op_matches = re.findall(tuple_pattern, ops_str)

        job_ops = []
        for machine_id, proc_time in op_matches:
            job_ops.append([int(machine_id), int(proc_time)])

        operations.append(job_ops)

    return operations


def extract_makespan_from_output(output_str):
    """Extract makespan from output string."""
    match = re.search(r'Makespan:\s*(\d+)', output_str)
    if match:
        return int(match.group(1))
    return None


def filter_and_convert_dataset(input_file, output_file, max_count, max_jobs=10, desc="Processing"):
    """Filter dataset and convert to new format."""
    print(f"\nLoading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Total records in source: {len(data)}")

    filtered = []
    skipped = 0

    for rec in tqdm(data, desc=desc):
        if len(filtered) >= max_count:
            break

        n = int(rec['n'])
        m = int(rec['m'])

        if n >= max_jobs:
            continue

        # Parse operations from input field
        operations = parse_operations_from_input(rec.get('input', ''), n, m)

        if len(operations) != n or any(len(ops) != m for ops in operations):
            skipped += 1
            continue

        # Extract makespan
        makespan = extract_makespan_from_output(rec['output'])

        filtered.append({
            'n_jobs': n,
            'n_machines': m,
            'operations': operations,
            'output': rec['output'],
            'optimal_makespan': makespan,
        })

    print(f"Filtered: {len(filtered)} instances (skipped {skipped} due to parsing errors)")

    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=2)

    print(f"Saved to {output_file}")
    return len(filtered)


def main():
    # Paths
    train_input = '/root/autodl-tmp/yao/VisionSolver-main-jssp/data/sft/train/training_jssp-001.json'
    test_input = '/root/autodl-tmp/yao/VisionSolver-main-jssp/data/sft/eval/test.json'

    train_output = '/root/autodl-tmp/yao/Hercules/problems/vision_jssp/train_jssp_5000_under30.json'
    test_output = '/root/autodl-tmp/yao/Hercules/problems/vision_jssp/test_jssp_50_under30.json'

    print("="*60)
    print("Extracting JSSP datasets (n_jobs < 10)")
    print("="*60)

    # Filter training data
    train_count = filter_and_convert_dataset(train_input, train_output, 5000, max_jobs=10, desc="Training")

    # Filter test data
    test_count = filter_and_convert_dataset(test_input, test_output, 50, max_jobs=10, desc="Test")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training: {train_count} instances")
    print(f"Test:     {test_count} instances")
    print(f"\nFiles saved:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")


if __name__ == '__main__':
    main()
