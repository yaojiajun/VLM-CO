def draw_mvc_v1_v2(edges: list, num_nodes: int, output_path: str) -> None:
    """
    Draw Minimum Vertex Cover graph with node boxes showing neighbors.
    Enhanced visualization for more intuitive MVC interpretation.

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

    # Fixed grid parameters
    grid_size = 10
    box_width = 1.4
    box_height_main = 0.5
    neighbor_box_size = 0.3
    main_fontsize = 16
    neighbor_fontsize = 12
    margin = 0.6
    cell_width = box_width + 0.4
    cell_height = box_height_main + 1.8

    fig_width = grid_size * cell_width + 2 * margin
    fig_height = grid_size * cell_height + 2 * margin + 1

    # Create NetworkX graph and positions
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Normalize position for grid mapping
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

    # Assign nodes to grid cells 
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

        best_cell = best_cell if best_cell else (grid_x, grid_y)
        occupied[best_cell] = node_id
        x = margin + best_cell[0] * cell_width + cell_width / 2
        y = margin + 1 + (grid_size - 1 - best_cell[1]) * cell_height + cell_height / 2

        node_positions[node_id] = (x, y)

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis('off')
    ax.set_aspect('equal')

    # Draw nodes with enhanced visual cues
    for node_id in range(num_nodes):
        x, y = node_positions[node_id]
        node_neighbors = neighbors.get(node_id, [])
        num_neighbor_rows = (len(node_neighbors) + neighbors_per_row - 1) // neighbors_per_row
        neighbor_row_height = num_neighbor_rows * (neighbor_box_size + 0.1) + 0.15
        total_box_height = box_height_main + neighbor_row_height

        degree_ratio = degrees[node_id] / max_degree if max_degree > 0 else 0

        # Enhanced color gradients for visual importance
        if degree_ratio > 0.7:
            box_color = '#FFC1C1'  # Light red for critical nodes
            edge_color = '#FF2D00'  # Dark red border
            text_color = '#780000'
        elif degree_ratio > 0.4:
            box_color = '#FFD9A0'  # Soft orange
            edge_color = '#FF8B00'  # Dim orange border
            text_color = '#783D00'
        else:
            box_color = '#C5D7FF'  # Soft blue for low importance
            edge_color = '#4D7FFF'  # Bright blue border
            text_color = '#0F2F68'

        # Draw approximated node boxes
        linewidth = 2.5 + degree_ratio * 1.0  # Scaling border width
        outer_box = patches.Rectangle(
            (x - box_width / 2, y - total_box_height / 2),
            box_width, total_box_height,
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=linewidth,
            zorder=2
        )
        ax.add_patch(outer_box)

        # Draw node ID
        text_y = y + (total_box_height / 2 - box_height_main / 2)
        ax.text(x, text_y, f"{node_id}",
                fontsize=main_fontsize, ha='center', va='center',
                color=text_color, fontweight='bold', zorder=3)

        # Draw separator
        line_y = y + (total_box_height / 2 - box_height_main)
        ax.plot([x - box_width / 2, x + box_width / 2], [line_y, line_y],
                color='#3B3B3B', linewidth=2.0, zorder=3)

        # Draw neighbor section
        neighbor_area_top = line_y - 0.1

        for row_idx in range(num_neighbor_rows):
            row_y = neighbor_area_top - (row_idx + 0.5) * (neighbor_box_size + 0.1)

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
                            color='#0A0A0A', fontweight='bold', zorder=3)

    # Title
    ax.text(fig_width / 2, fig_height - 0.4,
            f"MVC Visualization (n={num_nodes}, e={len(edges)})",
            fontsize=18, fontweight='bold', ha='center', va='top',
            color='#1F2937')

    plt.tight_layout(pad=1.0)
    plt.savefig(output_path, bbox_inches='tight', dpi=100, facecolor='white')
    plt.close()
