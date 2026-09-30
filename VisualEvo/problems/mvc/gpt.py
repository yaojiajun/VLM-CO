def draw_mvc_v1_v2(edges: list, num_nodes: int, output_path: str) -> None:
    """
    Draw Minimum Vertex Cover graph with node boxes showing neighbors.
    Uses enhanced visual cues and a grid layout.

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
    import math

    # Build neighbor adjacency list from edges
    neighbors = {}
    for a, b in edges:
        if a not in neighbors:
            neighbors[a] = []
        if b not in neighbors:
            neighbors[b] = []
        neighbors[a].append(b)
        neighbors[b].append(a)

    # Calculate node degrees for visual encoding
    degrees = {node: len(neighbors.get(node, [])) for node in range(num_nodes)}
    max_degree = max(degrees.values()) if degrees else 1

    # Adaptive grid size based on number of nodes with generous spacing
    target_cells = min(100, int(num_nodes * 1.4 + 30))
    grid_size = int(math.ceil(math.sqrt(target_cells)))
    box_width = 1.2
    box_height_main = 0.4
    neighbor_box_size = 0.25
    main_fontsize = 14
    neighbor_fontsize = 11
    margin = 0.5
    cell_width = box_width + 0.3
    cell_height = box_height_main + 1.5

    fig_width = grid_size * cell_width + 2 * margin
    fig_height = grid_size * cell_height + 2 * margin + 1

    # Create graph and compute node positions
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    
    # Initial layout using spring layout method
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Normalize positions to grid coordinates
    normalized_pos = {}
    for i in range(num_nodes):
        nx_val = (pos[i][0] - min(pos.values(), key=lambda x: x[0])[0]) / (max(pos.values(), key=lambda x: x[0])[0] - min(pos.values(), key=lambda x: x[0])[0]) or 1)
        ny_val = (pos[i][1] - min(pos.values(), key=lambda x: x[1])[1]) / (max(pos.values(), key=lambda x: x[1])[1] - min(pos.values(), key=lambda x: x[1])[1]) or 1)
        grid_x = int(nx_val * (grid_size - 1))
        grid_y = int(ny_val * (grid_size - 1))
        normalized_pos[i] = (grid_x, grid_y)

    # Assign nodes to grid cells while avoiding overlap
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

                if 0 <= cx < grid_size and 0 <= cy < grid_size and (cx, cy) not in occupied:
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

    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis('off')
    ax.set_aspect('equal')

    # Draw nodes with enhanced visual encoding for degree
    for node_id in range(num_nodes):
        x, y = node_positions[node_id]
        node_neighbors = neighbors.get(node_id, [])
        num_neighbor_rows = (len(node_neighbors) + neighbors_per_row - 1) // neighbors_per_row
        neighbor_row_height = num_neighbor_rows * (neighbor_box_size + 0.06) + 0.12
        total_box_height = box_height_main + neighbor_row_height

        # Degree-based color intensity and visual importance
        degree_ratio = degrees[node_id] / max_degree if max_degree > 0 else 0
        if degree_ratio > 0.7:
            box_color = '#FEE2E2'  # Light red
            edge_color = '#DC2626'
            text_color = '#991B1B'
        elif degree_ratio > 0.4:
            box_color = '#FED7AA'  # Light orange
            edge_color = '#EA580C'
            text_color = '#9A3412'
        else:
            box_color = '#E0E7FF'  # Light blue
            edge_color = '#4F46E5'
            text_color = '#3730A3'

        # Draw the outer box with enhanced aesthetics
        linewidth = 2.0 + degree_ratio * 1.5
        outer_box = patches.Rectangle(
            (x - box_width / 2, y - total_box_height / 2),
            box_width, total_box_height,
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=linewidth,
            zorder=2
        )
        ax.add_patch(outer_box)

        # Node ID display
        text_y = y + (total_box_height / 2 - box_height_main / 2)
        ax.text(x, text_y, f"{node_id}",
                fontsize=main_fontsize, ha='center', va='center',
                color=text_color, fontweight='bold', zorder=3)

        # Separator line
        line_y = y + (total_box_height / 2 - box_height_main)
        ax.plot([x - box_width / 2, x + box_width / 2], [line_y, line_y],
                color='#6B7280', linewidth=1.5, zorder=3)

        # Neighbor section
        neighbor_area_top = line_y - 0.08
        for row_idx in range(num_neighbor_rows):
            row_y = neighbor_area_top - (row_idx + 0.5) * (neighbor_box_size + 0.06)

            start_idx = row_idx * neighbors_per_row
            end_idx = min(start_idx + neighbors_per_row, len(node_neighbors))
            row_neighbors = node_neighbors[start_idx:end_idx]

            col_width = box_width / 4
            for col_idx in range(4):
                col_x = x - box_width / 2 + col_idx * col_width + col_width / 2
                if col_idx < len(row_neighbors):
                    neighbor_id = row_neighbors[col_idx]
                    ax.text(col_x, row_y, str(neighbor_id),
                            fontsize=neighbor_fontsize, ha='center', va='center',
                            color='#111827', fontweight='bold', zorder=3)

    # Title with additional info
    ax.text(fig_width / 2, fig_height - 0.3,
            f"MVC (n={num_nodes}, e={len(edges)}, grid={grid_size}×{grid_size})",
            fontsize=16, fontweight='bold', ha='center', va='top',
            color='#1F2937')

    plt.tight_layout(pad=0.8)
    plt.savefig(output_path, bbox_inches='tight', dpi=100, facecolor='white')
    plt.close()
