"""
Utility functions for MVC evaluation.
"""

import re
from typing import List, Tuple


def parse_mvc_output(text: str) -> Tuple[List[int], int]:
    """
    Extract set and objective from MVC output.

    Expected format: "Set: [1, 2, 3], Objective: 3"

    Returns:
        (node_set, objective): List of node IDs and objective value
    """
    set_match = re.search(r'Set:\s*\[([^\]]*)\]', text)
    obj_match = re.search(r'Objective:\s*(\d+)', text)

    node_set = []
    if set_match:
        ids_str = set_match.group(1).strip()
        if ids_str:
            try:
                node_set = [int(x.strip()) for x in ids_str.split(',') if x.strip()]
            except ValueError:
                node_set = []

    objective = int(obj_match.group(1)) if obj_match else 0

    return node_set, objective


def parse_mis_output(text: str) -> Tuple[List[int], int]:
    """
    Extract set and objective from MIS output.

    Expected format: "Set: [1, 2, 3], Objective: 3"

    Returns:
        (node_set, objective): List of node IDs and objective value
    """
    set_match = re.search(r'Set:\s*\[([^\]]*)\]', text)
    obj_match = re.search(r'Objective:\s*(\d+)', text)

    node_set = []
    if set_match:
        ids_str = set_match.group(1).strip()
        if ids_str:
            try:
                node_set = [int(x.strip()) for x in ids_str.split(',') if x.strip()]
            except ValueError:
                node_set = []

    objective = int(obj_match.group(1)) if obj_match else 0

    return node_set, objective


def is_vertex_cover(node_set: List[int], edges: List[Tuple[int, int]]) -> bool:
    """
    Check if a set of nodes forms a valid vertex cover.

    Args:
        node_set: List of node IDs
        edges: List of edges (node_a, node_b)

    Returns:
        True if every edge has at least one endpoint in node_set
    """
    node_set_set = set(node_set)
    for a, b in edges:
        if a not in node_set_set and b not in node_set_set:
            return False
    return True


def is_independent_set(node_set: List[int], edges: List[Tuple[int, int]]) -> bool:
    """
    Check if a set of nodes forms an independent set.

    Args:
        node_set: List of node IDs
        edges: List of edges (node_a, node_b)

    Returns:
        True if no two nodes in node_set are connected by an edge
    """
    node_set_set = set(node_set)
    for a, b in edges:
        if a in node_set_set and b in node_set_set:
            return False
    return True


def compute_metric_mvc(predictions: List[str],
                       labels: List[str],
                       instances: List) -> Tuple[float, float]:
    """
    Compute metrics for Minimum Vertex Cover predictions.

    Args:
        predictions: List of prediction strings
        labels: List of ground truth output strings
        instances: List of [num_nodes, edges] for each instance

    Returns:
        (exact_accuracy, objective_accuracy): Tuple of accuracy metrics
    """
    exact_match = 0
    obj_match = 0
    total = len(predictions)

    for pred_text, label_text, instance in zip(predictions, labels, instances):
        pred_set, pred_obj = parse_mvc_output(pred_text)
        true_set, true_obj = parse_mvc_output(label_text)

        # Check exact match
        if sorted(pred_set) == sorted(true_set):
            exact_match += 1

        # Check objective match
        if pred_obj == true_obj:
            obj_match += 1

    exact_acc = exact_match / total if total > 0 else 0.0
    obj_acc = obj_match / total if total > 0 else 0.0

    return exact_acc, obj_acc


def compute_metric_mis(predictions: List[str],
                       labels: List[str],
                       instances: List) -> Tuple[float, float]:
    """
    Compute metrics for Maximum Independent Set predictions.

    Args:
        predictions: List of prediction strings
        labels: List of ground truth output strings
        instances: List of [num_nodes, edges] for each instance

    Returns:
        (exact_accuracy, objective_accuracy): Tuple of accuracy metrics
    """
    exact_match = 0
    obj_match = 0
    total = len(predictions)

    for pred_text, label_text, instance in zip(predictions, labels, instances):
        pred_set, pred_obj = parse_mis_output(pred_text)
        true_set, true_obj = parse_mis_output(label_text)

        # Check exact match
        if sorted(pred_set) == sorted(true_set):
            exact_match += 1

        # Check objective match
        if pred_obj == true_obj:
            obj_match += 1

    exact_acc = exact_match / total if total > 0 else 0.0
    obj_acc = obj_match / total if total > 0 else 0.0

    return exact_acc, obj_acc
