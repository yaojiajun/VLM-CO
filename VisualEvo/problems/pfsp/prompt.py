def get_instruction_v2(n_jobs: int, n_machines: int) -> str:
    """
    V-PFSP instruction following the paper specification.
    Based on: Vision-Language Models as Combinatorial Optimization Solvers (ICLR 2027)
    MUST match training format exactly.
    """
    return (
        f"This is a Permutation Flow Shop Problem (PFSP) instance with {n_jobs} jobs and {n_machines} "
        f"machines. The image shows a processing time matrix where each row represents a machine "
        f"(M0, M1, ..., M{n_machines-1}) and each column represents a job (J0, J1, ..., J{n_jobs-1}). Each cell is displayed as "
        f"a colored box containing the processing time (t=value) for that job on that machine, where "
        f"different jobs are represented by distinct colors for visual distinction. All jobs must be "
        f"processed on each machine in the same order. Each machine can process only one job at a "
        f"time, and each job can be processed by only one machine at a time. The goal is to find the job "
        f"processing order that minimizes the makespan (maximum completion time). Output format — "
        f"no other text: Order: [job sequence], Objective: <makespan_value>"
    )
        f"Provide the solution in the following format:\n\n"
        f"1. Schedule: List the job sequence (0-indexed).\n"
        f"2. Makespan: The makespan of the schedule."
    )
