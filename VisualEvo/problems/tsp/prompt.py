def get_instruction_v1(num_nodes: int) -> str:
    """
    V-TSP instruction following the paper specification.
    Based on: Vision-Language Models as Combinatorial Optimization Solvers (ICLR 2027)
    """
    return (
        f"This is a TSP instance with {num_nodes} cities, where the starting city (node 0) is marked "
        f"with a star symbol ★ and the other cities (nodes 1–{num_nodes-1}) are represented as circles colored by "
        f"spatial cluster with their node IDs labeled above. The goal is to find a tour starting from node "
        f"0, visiting all {num_nodes} cities exactly once, and returning to node 0 while minimizing the total travel "
        f"distance. Output format — no other text: Route: [0, n1, n2, ..., 0], Objective: <total_distance>"
    )
