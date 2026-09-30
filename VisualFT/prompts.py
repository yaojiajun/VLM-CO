"""
prompts.py
==========
Instruction builders for all 6 combinatorial optimization problems.
Based on: Vision-Language Models as Combinatorial Optimization Solvers (ICLR 2027)

Each instruction builder follows the paper specification for V-TSP, V-CVRP,
V-MIS, V-MVC, V-JSSP, and V-PFSP.
"""


def build_tsp_instruction(num_nodes: int, **kwargs) -> str:
    """
    V-TSP instruction following the paper specification.

    Args:
        num_nodes: Number of cities in the TSP instance

    Returns:
        Instruction string for the VLM
    """
    return (
        f"This is a TSP instance with {num_nodes} cities, where the starting city (node 0) is marked "
        f"with a star symbol ★ and the other cities (nodes 1–{num_nodes-1}) are represented as circles colored by "
        f"spatial cluster with their node IDs labeled above. The goal is to find a tour starting from node "
        f"0, visiting all {num_nodes} cities exactly once, and returning to node 0 while minimizing the total travel "
        f"distance. Output format — no other text: Route: [0, n1, n2, ..., 0], Objective: <total_distance>"
    )


def build_cvrp_instruction(vehicle_capacity: str, num_nodes: int, **kwargs) -> str:
    """
    V-CVRP instruction following the paper specification.

    Args:
        vehicle_capacity: Maximum capacity of each vehicle
        num_nodes: Total number of nodes (depot + customers)

    Returns:
        Instruction string for the VLM
    """
    cap = int(float(vehicle_capacity))
    n_customers = int(num_nodes) - 1
    return (
        f"This is a CVRP instance with 1 depot and {n_customers} customer nodes. Each blue circle "
        f"represents a customer (labeled with node ID and demand in brackets), and the red star marks "
        f"the depot (node 0). The circle size is proportional to customer demand. The vehicle capacity "
        f"is {cap} units. Each route must start and end at the depot, visit customers assigned to it "
        f"exactly once, and ensure the total demand on its route does not exceed the capacity. The goal "
        f"is to assign all customers to vehicle routes such that the total travel distance is minimized. "
        f"Output format — no other text: Routes: [[0, ..., 0], [0, ..., 0], ...], Objective: <total_distance>"
    )


def build_mis_instruction(num_nodes: int, **kwargs) -> str:
    """
    V-MIS instruction following the paper specification.

    Args:
        num_nodes: Number of nodes in the graph

    Returns:
        Instruction string for the VLM
    """
    return (
        f"This is a Maximum Independent Set (MIS) instance with {num_nodes} nodes. Each node "
        f"is displayed as a box with its ID in the top section and the IDs of its adjacent (neighbor) nodes "
        f"listed in the bottom section (up to 4 neighbor IDs per row). Each node is colored based on "
        f"its degree, where darker colors indicate higher degree (more connections) and lighter colors "
        f"indicate lower degree (fewer connections). The goal is to find the largest subset of vertices "
        f"such that no two vertices in the subset are connected by an edge. Output format — no other "
        f"text: Set: [list of vertex IDs], Objective: <set_size>"
    )


def build_mvc_instruction(num_nodes: int, **kwargs) -> str:
    """
    V-MVC instruction following the paper specification.

    Args:
        num_nodes: Number of nodes in the graph

    Returns:
        Instruction string for the VLM
    """
    return (
        f"This is a Minimum Vertex Cover (MVC) instance with {num_nodes} nodes. Each node is "
        f"displayed as a box with its ID in the top section and the IDs of its adjacent (neighbor) nodes "
        f"listed in the bottom section (up to 4 neighbor IDs per row). Each node is colored based on "
        f"its degree, where darker colors indicate higher degree (more connections) and lighter colors "
        f"indicate lower degree (fewer connections). The goal is to find the smallest subset of vertices "
        f"such that every edge in the graph has at least one endpoint in the subset. Output format — no "
        f"other text: Set: [list of vertex IDs], Objective: <set_size>"
    )


def build_jssp_instruction(n_jobs: int, n_machines: int, **kwargs) -> str:
    """
    V-JSSP instruction following the paper specification.

    Args:
        n_jobs: Number of jobs
        n_machines: Number of machines

    Returns:
        Instruction string for the VLM
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


def build_pfsp_instruction(n_jobs: int, n_machines: int, **kwargs) -> str:
    """
    V-PFSP instruction following the paper specification.

    Args:
        n_jobs: Number of jobs
        n_machines: Number of machines

    Returns:
        Instruction string for the VLM
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


# Mapping from problem type to instruction builder
INSTRUCTION_BUILDERS = {
    'tsp': build_tsp_instruction,
    'cvrp': build_cvrp_instruction,
    'mis': build_mis_instruction,
    'mvc': build_mvc_instruction,
    'jssp': build_jssp_instruction,
    'pfsp': build_pfsp_instruction,
}


def get_instruction(problem_type: str, **kwargs) -> str:
    """
    Get instruction for a specific problem type.

    Args:
        problem_type: One of 'tsp', 'cvrp', 'mis', 'mvc', 'jssp', 'pfsp'
        **kwargs: Problem-specific parameters (num_nodes, vehicle_capacity, etc.)

    Returns:
        Instruction string

    Raises:
        ValueError: If problem_type is not recognized
    """
    if problem_type not in INSTRUCTION_BUILDERS:
        raise ValueError(
            f"Unknown problem type: {problem_type}. "
            f"Must be one of {list(INSTRUCTION_BUILDERS.keys())}"
        )

    return INSTRUCTION_BUILDERS[problem_type](**kwargs)
