def get_instruction_v2(num_nodes: int) -> str:
    """
    V-MIS instruction following the paper specification.
    Based on: Vision-Language Models as Combinatorial Optimization Solvers (ICLR 2027)

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
