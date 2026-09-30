#!/root/autodl-tmp/yao/llm_env/bin/python
"""
generate_jssp_data.py

Generates JSSP instances and solves them using OR-Tools to create
training and test datasets for the vision_jssp problem.

Usage:
    python generate_jssp_data.py --output train_jssp_first5000.json --count 5000 --n_jobs 6 --n_machines 6
    python generate_jssp_data.py --output test_jssp.json --count 100 --n_jobs 6 --n_machines 6
"""

import argparse
import json
import numpy as np
import sys
from ortools.sat.python import cp_model

sys.path.insert(0, '/root/autodl-tmp/yao/VisionSolver-main')
from Envs.JSSPEnv.JSSPEnv import get_makespan


def generate_jssp_instance(n_jobs, n_machines, seed=None):
    """
    Generate a random JSSP instance.

    Returns:
        operations: list of lists, operations[job][op] = (machine_id, processing_time)
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
            proc_time = np.random.randint(1, 100)
            job_ops.append([int(machine_id), int(proc_time)])
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

    # Variables
    # task_starts[job][op] = start time
    # task_ends[job][op] = end time
    # task_intervals[job][op] = interval variable
    horizon = sum(operations[j][op][1] for j in range(n_jobs) for op in range(n_machines))

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

    # Precedence constraints: operations of same job must execute in order
    for job in range(n_jobs):
        for op in range(n_machines - 1):
            model.Add(task_ends[job][op] <= task_starts[job][op + 1])

    # No overlap constraints: operations on same machine cannot overlap
    machine_to_intervals = {}
    for job in range(n_jobs):
        for op in range(n_machines):
            machine_id = operations[job][op][0]
            if machine_id not in machine_to_intervals:
                machine_to_intervals[machine_id] = []
            machine_to_intervals[machine_id].append((job, op, task_intervals[job][op]))

    for machine_id, intervals_list in machine_to_intervals.items():
        model.AddNoOverlap([interval for _, _, interval in intervals_list])

    # Objective: minimize makespan
    makespan_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan_var, [task_ends[job][-1] for job in range(n_jobs)])
    model.Minimize(makespan_var)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        makespan = solver.Value(makespan_var)

        # Extract schedule: for each machine, get the job sequence
        schedule = [[] for _ in range(n_machines)]
        machine_schedules = {}

        for job in range(n_jobs):
            for op in range(n_machines):
                machine_id = operations[job][op][0]
                start_time = solver.Value(task_starts[job][op])
                if machine_id not in machine_schedules:
                    machine_schedules[machine_id] = []
                machine_schedules[machine_id].append((start_time, job))

        # Sort by start time and extract job sequence
        for machine_id in range(n_machines):
            if machine_id in machine_schedules:
                machine_schedules[machine_id].sort()
                schedule[machine_id] = [job for _, job in machine_schedules[machine_id]]

        return schedule, makespan

    return None, None


def format_output(schedule, makespan):
    """Format schedule and makespan as expected output string."""
    return f"Schedule: {schedule}, Makespan: {makespan}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--count', type=int, default=100, help='Number of instances to generate')
    parser.add_argument('--n_jobs', type=int, default=6, help='Number of jobs')
    parser.add_argument('--n_machines', type=int, default=6, help='Number of machines')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--solver_time_limit', type=int, default=60, help='Solver time limit per instance (seconds)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    data = []
    solved_count = 0

    print(f"Generating {args.count} JSSP instances with {args.n_jobs} jobs and {args.n_machines} machines...")

    for i in range(args.count):
        # Generate instance
        operations = generate_jssp_instance(args.n_jobs, args.n_machines, seed=args.seed + i)

        # Solve with OR-Tools
        schedule, makespan = solve_jssp_ortools(operations, time_limit_seconds=args.solver_time_limit)

        if schedule is not None:
            output_str = format_output(schedule, makespan)

            data.append({
                'n_jobs': args.n_jobs,
                'n_machines': args.n_machines,
                'operations': operations,
                'output': output_str,
                'optimal_makespan': makespan,
            })
            solved_count += 1

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{args.count} instances ({solved_count} solved)")
        else:
            print(f"  Warning: Could not solve instance {i}")

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nDone! Saved {len(data)} instances to {args.output}")
    print(f"Success rate: {solved_count}/{args.count} ({100*solved_count/args.count:.1f}%)")


if __name__ == '__main__':
    main()
