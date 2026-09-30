def get_instruction_v2(n_jobs: int, n_machines: int) -> str:
    """
    V-JSSP instruction following the paper specification.
    Based on: Vision-Language Models as Combinatorial Optimization Solvers (ICLR 2027)
    MUST match main_train_vision_jssp.py exactly.
    """
    return (
        f"This is a Job Shop Scheduling Problem (JSSP) instance with {n_jobs} jobs and {n_machines} "
        f"machines. Each row represents a job (J0 to J{n_jobs-1}) and each column represents an operation "
        f"step. Each cell displays the machine ID (e.g., M2) and processing time (e.g., t=56) for that "
        f"operation, with different machines color-coded. Arrows indicate that operations within a job "
        f"must be processed strictly from left to right. Each machine can process only one operation at "
        f"a time. The goal is to find a schedule that determines the completion time of all jobs while "
        f"minimizing the makespan (maximum completion time). Output format — no other text: Schedule: [[job order on M0], [job order on M1], ...], "
        f"Makespan: <makespan_value>"
    )
