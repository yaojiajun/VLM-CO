import numpy as np
def draw_v2(edges: list, num_nodes: int, output_path: str) -> None:
    """
    Draw Maximum Independent Set graph with node boxes showing neighbors.

    Args:
        edges: List of tuples (node_a, node_b) representing edges
        num_nodes: Number of nodes in the graph
        output_path: Path to save the PNG image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import networkx as nx
    import numpy as np

    # Build neighbor adjacency list from edges (avoid duplicates)
    neighbors = {}
    for a, b in edges:
        if a not in neighbors:
            neighbors[a] = []
        if b not in neighbors:
            neighbors[b] = []
        if b not in neighbors[a]:
            neighbors[a].append(b)
        if a not in neighbors[b]:
            neighbors[b].append(a)

    # Adaptive grid size based on number of nodes
    import math
    if num_nodes <= 10:
        target_cells = 40
    elif num_nodes <= 20:
        target_cells = 45
    elif num_nodes <= 50:
        target_cells = 70
    else:
        target_cells = min(100, int(num_nodes * 1.4))
    
    grid_size = int(math.ceil(math.sqrt(target_cells)))
    
    box_size = 1.2
    box_height_main = 0.5
    neighbor_box_size = 0.3
    main_fontsize = 16
    neighbor_fontsize = 12
    margin = 0.6
    k_value = 2.5

    cell_width = box_size + 0.3
    cell_height = box_height_main + 1.3

    fig_width = grid_size * cell_width + 2 * margin
    fig_height = grid_size * cell_height + 2 * margin + 1

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, k=k_value, iterations=100, seed=42)
    
    xs = [pos[i][0] for i in range(num_nodes)]
    ys = [pos[i][1] for i in range(num_nodes)]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    normalized_pos = {}
    for i in range(num_nodes):
        nx_val = (pos[i][0] - x_min) / (x_max - x_min) if x_max != x_min else 0.5
        ny_val = (pos[i][1] - y_min) / (y_max - y_min) if y_max != y_min else 0.5
        grid_x = int(nx_val * (grid_size - 1))
        grid_y = int(ny_val * (grid_size - 1))
        normalized_pos[i] = (grid_x, grid_y)

    occupied = {}
    node_positions = {}
    neighbors_per_row = 4

    for node_id in range(num_nodes):
        grid_x, grid_y = normalized_pos[node_id]
        best_cell = None
        best_dist = float('inf')

        for dx in range(-grid_size, grid_size + 1):
            for dy in range(-grid_size, grid_size + 1):
                cx = grid_x + dx
                cy = grid_y + dy

                if 0 <= cx < grid_size and 0 <= cy < grid_size:
                    if (cx, cy) not in occupied:
                        dist = dx * dx + dy * dy
                        if dist < best_dist:
                            best_dist = dist
                            best_cell = (cx, cy)

        if best_cell is None:
            best_cell = (grid_x, grid_y)

        occupied[best_cell] = node_id
        x = margin + best_cell[0] * cell_width + cell_width / 2
        y = margin + 1 + (grid_size - 1 - best_cell[1]) * cell_height + cell_height / 2
        node_positions[node_id] = (x, y)

    degrees = {node_id: len(neighbors.get(node_id, [])) for node_id in range(num_nodes)}
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0

    def get_node_color(degree):
        if max_degree == min_degree:
            return '#AED581'  # Light green for uniform degree
        norm_degree = (degree - min_degree) / (max_degree - min_degree)
        r_light, g_light, b_light = 174, 229, 129
        r_dark, g_dark, b_dark = 56, 142, 60
        r = int(r_light + (r_dark - r_light) * norm_degree)
        g = int(g_light + (g_dark - g_light) * norm_degree)
        b = int(b_light + (b_dark - b_light) * norm_degree)
        return f'#{r:02x}{g:02x}{b:02x}'

    node_box_heights = {}
    for node_id in range(num_nodes):
        node_neighbors = neighbors.get(node_id, [])
        num_neighbor_rows = (len(node_neighbors) + neighbors_per_row - 1) // neighbors_per_row
        neighbor_row_height = num_neighbor_rows * (neighbor_box_size + 0.08) + 0.12
        node_box_heights[node_id] = box_height_main + neighbor_row_height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis('off')
    ax.set_aspect('equal')

    for node_id in range(num_nodes):
        x, y = node_positions[node_id]
        node_neighbors = neighbors.get(node_id, [])
        num_neighbor_rows = (len(node_neighbors) + neighbors_per_row - 1) // neighbors_per_row
        neighbor_row_height = num_neighbor_rows * (neighbor_box_size + 0.08) + 0.12
        total_box_height = box_height_main + neighbor_row_height
        
        node_degree = degrees[node_id]
        box_color = get_node_color(node_degree)
        
        outer_box = patches.Rectangle(
            (x - box_size / 2, y - total_box_height / 2),
            box_size, total_box_height,
            facecolor=box_color,
            edgecolor='#43A047',
            linewidth=3,
            zorder=2,
            alpha=0.85
        )
        ax.add_patch(outer_box)

        text_y = y + (total_box_height / 2 - box_height_main / 2)
        ax.text(x, text_y, str(node_id),
                fontsize=main_fontsize, ha='center', va='center',
                color='#1B5E20', fontweight='bold', zorder=3)

        line_y = y + (total_box_height / 2 - box_height_main)
        ax.plot([x - box_size / 2, x + box_size / 2], [line_y, line_y],
                color='#BDBDBD', linewidth=2, zorder=3)

        neighbor_area_top = line_y - 0.08

        for row_idx in range(num_neighbor_rows):
            row_y = neighbor_area_top - (row_idx + 0.5) * (neighbor_box_size + 0.08)

            start_idx = row_idx * neighbors_per_row
            end_idx = min(start_idx + neighbors_per_row, len(node_neighbors))
            row_neighbors = node_neighbors[start_idx:end_idx]

            col_width = box_size / 4

            for col_idx in range(4):
                col_x = x - box_size / 2 + col_idx * col_width + col_width / 2

                if col_idx < len(row_neighbors):
                    neighbor_id = row_neighbors[col_idx]
                    ax.text(col_x, row_y, str(neighbor_id),
                            fontsize=neighbor_fontsize, ha='center', va='center',
                            color='#33691E', fontweight='bold', zorder=3)

    ax.text(fig_width / 2, fig_height - 0.3,
            f"Maximum Independent Set Visualization (n={num_nodes})",
            fontsize=18, fontweight='bold', ha='center', va='top',
            color='#1A237E')

    plt.tight_layout(pad=0.8)
    plt.savefig(output_path, bbox_inches='tight', dpi=100, facecolor='white')
    plt.close()
