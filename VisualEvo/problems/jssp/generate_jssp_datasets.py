#!/usr/bin/env python3
"""
generate_jssp_datasets.py

Generates JSSP training and test datasets with OR-Tools solutions.
- Training: 5000 instances with n*m < 30 (e.g., 6x5, 5x5, 4x7, etc.)
- Test: 50 instances with n*m < 30
"""

import argparse
import json
import numpy as np
import sys
from ortools.sat.python import cp_model
from tqdm import tqdm


def generate_jssp_instance(n_jobs, n_machines, seed=None):
    """
    Generate a random JSSP instance.

    Returns:
        operations: list of lists, operations[job][op] = [machine_id, processing_time]
    """
    if seed is not None:
        np.random.seed(seed)

    operations = []
    for job in range(n_jobs):
        # Each job visits each machine exactly once, in a random order
        machine_order = np.random.permutation(n_machines).tolist()
        job_ops = []
        for machine_id in machine_order:
            # Random processing time between 1 and 99
            proc_time = int(np.random.randint(1, 100))
            job_ops.append([int(machine_id), proc_time])
        operations.append(job_ops)

    return operations


def solve_jssp_ortools(operations, time_limit_seconds=60):
    """
    Solve JSSP using OR-Tools CP-SAT solver.

    Returns:
        (schedule, makespan) or (None, None) if failed
        schedule: list of lists, schedule[machine] = [job_ids in order]
    """
    n_jobs = len(operations)
    n_machines = len(operations[0])

    model = cp_model.CpModel()

    # Calculate horizon
    horizon = sum(operations[j][op][1] for j in range(n_jobs) for op in range(n_machines))

    # Variables
    task_starts = []
    task_ends = []
    task_intervals = []

    for job in range(n_jobs):
        job_starts = []
        job_ends = []
        job_intervals = []
        for op in range(n_machines):
            machine_id, proc_time = operations[job][op]
            start_var = model.NewIntVar(0, horizon, f'start_j{job}_o{op}')
            end_var = model.NewIntVar(0, horizon, f'end_j{job}_o{op}')
            interval_var = model.NewIntervalVar(start_var, proc_time, end_var, f'interval_j{job}_o{op}')
            job_starts.append(start_var)
            job_ends.append(end_var)
            job_intervals.append(interval_var)
        task_starts.append(job_starts)
        task_ends.append(job_ends)
        task_intervals.append(job_intervals)

    # Precedence constraints
    for job in range(n_jobs):
        for op in range(n_machines - 1):
            model.Add(task_ends[job][op] <= task_starts[job][op + 1])

    # No overlap constraints
    machine_to_intervals = {}
    for job in range(n_jobs):
        for op in range(n_machines):
            machine_id = operations[job][op][0]
            if machine_id not in machine_to_intervals:
                machine_to_intervals[machine_id] = []
            machine_to_intervals[machine_id].append((job, op, task_intervals[job][op]))

    for machine_id, intervals_list in machine_to_intervals.items():
        model.AddNoOverlap([interval for _, _, interval in intervals_list])

    # Objective
    makespan_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan_var, [task_ends[job][-1] for job in range(n_jobs)])
    model.Minimize(makespan_var)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        makespan = solver.Value(makespan_var)

        # Extract schedule
        schedule = [[] for _ in range(n_machines)]
        machine_schedules = {}

        for job in range(n_jobs):
            for op in range(n_machines):
                machine_id = operations[job][op][0]
                start_time = solver.Value(task_starts[job][op])
                if machine_id not in machine_schedules:
                    machine_schedules[machine_id] = []
                machine_schedules[machine_id].append((start_time, job))

        # Sort by start time
        for machine_id in range(n_machines):
            if machine_id in machine_schedules:
                machine_schedules[machine_id].sort()
                schedule[machine_id] = [job for _, job in machine_schedules[machine_id]]

        return schedule, makespan

    return None, None


def format_output(schedule, makespan):
    """Format schedule and makespan as expected output string."""
    return f"Schedule: {schedule}, Makespan: {makespan}"


def generate_problem_sizes(max_product=30, count=5000):
    """Generate diverse problem sizes with n*m < max_product."""
    sizes = []
    # Common sizes
    base_sizes = [
        (5, 5), (6, 5), (5, 6), (6, 4), (4, 6),
        (7, 4), (4, 7), (3, 8), (8, 3), (3, 9), (9, 3),
    ]

    # Generate instances
    for i in range(count):
        n_jobs, n_machines = base_sizes[i % len(base_sizes)]
        if n_jobs * n_machines < max_product:
            sizes.append((n_jobs, n_machines))

    return sizes[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_train', type=str, default='train_jssp_5000_under30.json')
    parser.add_argument('--output_test', type=str, default='test_jssp_50_under30.json')
    parser.add_argument('--train_count', type=int, default=5000)
    parser.add_argument('--test_count', type=int, default=50)
    parser.add_argument('--max_product', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--solver_time_limit', type=int, default=60)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Generate training data
    print(f"Generating {args.train_count} training instances (n*m < {args.max_product})...")
    train_sizes = generate_problem_sizes(args.max_product, args.train_count)
    train_data = []
    train_solved = 0

    for i, (n_jobs, n_machines) in enumerate(tqdm(train_sizes, desc="Training")):
        operations = generate_jssp_instance(n_jobs, n_machines, seed=args.seed + i)
        schedule, makespan = solve_jssp_ortools(operations, time_limit_seconds=args.solver_time_limit)

        if schedule is not None:
            output_str = format_output(schedule, makespan)
            train_data.append({
                'n_jobs': n_jobs,
                'n_machines': n_machines,
                'operations': operations,
                'output': output_str,
                'optimal_makespan': makespan,
            })
            train_solved += 1
        else:
            print(f"  Warning: Could not solve training instance {i} ({n_jobs}x{n_machines})")

    # Generate test data
    print(f"\nGenerating {args.test_count} test instances (n*m < {args.max_product})...")
    test_sizes = generate_problem_sizes(args.max_product, args.test_count)
    test_data = []
    test_solved = 0

    for i, (n_jobs, n_machines) in enumerate(tqdm(test_sizes, desc="Test")):
        operations = generate_jssp_instance(n_jobs, n_machines, seed=args.seed + args.train_count + i)
        schedule, makespan = solve_jssp_ortools(operations, time_limit_seconds=args.solver_time_limit)

        if schedule is not None:
            output_str = format_output(schedule, makespan)
            test_data.append({
                'n_jobs': n_jobs,
                'n_machines': n_machines,
                'operations': operations,
                'output': output_str,
                'optimal_makespan': makespan,
            })
            test_solved += 1
        else:
            print(f"  Warning: Could not solve test instance {i} ({n_jobs}x{n_machines})")

    # Save to JSON
    print(f"\nSaving training data to {args.output_train}...")
    with open(args.output_train, 'w') as f:
        json.dump(train_data, f, indent=2)

    print(f"Saving test data to {args.output_test}...")
    with open(args.output_test, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training: {len(train_data)}/{args.train_count} instances solved ({100*train_solved/args.train_count:.1f}%)")
    print(f"Test:     {len(test_data)}/{args.test_count} instances solved ({100*test_solved/args.test_count:.1f}%)")
    print(f"\nFiles saved:")
    print(f"  - {args.output_train}")
    print(f"  - {args.output_test}")


if __name__ == '__main__':
    main()
