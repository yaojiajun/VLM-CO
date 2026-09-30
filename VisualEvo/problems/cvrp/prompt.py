def get_instruction_v2(vehicle_capacity: str, num_nodes: int) -> str:
    """
    V-CVRP instruction following the paper specification.
    Based on: Vision-Language Models as Combinatorial Optimization Solvers (ICLR 2027)
    """
    return (
        f"This is a CVRP instance with 1 depot and {num_nodes} customer nodes. Each blue circle "
        f"represents a customer (labeled with node ID and demand in brackets), and the red star marks "
        f"the depot (node 0). The circle size is proportional to customer demand. The vehicle capacity "
        f"is {vehicle_capacity} units. Each route must start and end at the depot, visit customers assigned to it "
        f"exactly once, and ensure the total demand on its route does not exceed the capacity. The goal "
        f"is to assign all customers to vehicle routes such that the total travel distance is minimized. "
        f"Output format — no other text: Routes: [[0, ..., 0], [0, ..., 0], ...], Objective: <total_distance>"
    )