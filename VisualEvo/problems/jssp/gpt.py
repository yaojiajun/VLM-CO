def draw_v2(instance: dict, out_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    operations = instance['operations']

    # Advanced color scheme
    COLORS = [
        '#A5C8E1',  # Custom Light Blue
        '#FCA3A3',  # Custom Light Red
        '#BBE1CB',  # Custom Light Green
        '#F3ED9E',  # Custom Light Yellow
        '#F6A5D0',  # Custom Light Pink
        '#B0B8CA',  # Custom Light Grey Blue
        '#F1A7B1',  # Custom Soft Pink
        '#C3E6D1',  # Custom Soft Green
        '#F7A9BA',  # Custom Soft Orange
        '#B2A6D2',  # Custom Soft Purple
        '#FEC289',  # Custom Soft Yellow Orange
        '#68E4E9',  # Custom Soft Cyan
    ]

    scale = 0.7
    cell_width = 1.0 * scale
    cell_height = 0.9 * scale
    spacing_x = 0.2 * scale
    spacing_y = 0.2 * scale

    fig_width = max(10, 1.0 + n_machines * (cell_width + spacing_x))
    fig_height = 1.5 + n_jobs * (cell_height + spacing_y)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('#F0F0F0')
    ax.set_facecolor('#F0F0F0')

    title_y = n_jobs * (cell_height + spacing_y) + 1.2
    ax.text(n_machines * (cell_width + spacing_x) / 2, title_y + 0.3,
            f'Job Shop Scheduling Problem Visualization: {n_jobs} Jobs × {n_machines} Machines',
            ha='center', va='top', fontsize=16, fontweight='bold', color='#333333')
    ax.text(n_machines * (cell_width + spacing_x) / 2, title_y,
            'OpX (M_ID; elapsed_time)',
            ha='center', va='top', fontsize=10, style='italic', color='#666666')

    for op in range(n_machines):
        x = op * (cell_width + spacing_x) + cell_width / 2
        y = title_y - 0.7
        ax.text(x, y, f'Op{op + 1}', ha='center', va='center',
                fontsize=12, fontweight='bold', color='#202020')
                
        if op < n_machines - 1:
            arrow_x = x + cell_width / 2 + 0.05
            ax.annotate('', xy=(arrow_x + 0.05, y), xytext=(arrow_x, y),
                       arrowprops=dict(arrowstyle='->', color='#888888', lw=1.8))

    for job_idx in range(n_jobs):
        job_ops = operations[job_idx]
        y = title_y - 1.5 - job_idx * (cell_height + spacing_y)

        ax.text(-0.6, y - cell_height / 2, f'Job {job_idx + 1}',
                ha='right', va='center', fontsize=12, fontweight='bold', color='#333333')

        for op_idx in range(len(job_ops)):
            machine, time = job_ops[op_idx]
            x = op_idx * (cell_width + spacing_x)

            color = COLORS[job_idx % len(COLORS)]
            rect = patches.Rectangle((x, y - cell_height), cell_width, cell_height,
                                     linewidth=2, edgecolor='#000000',
                                     facecolor=color, alpha=0.85)
            ax.add_patch(rect)

            # Adding a border around each operation
            rect_outline = patches.Rectangle(
                (x, y - cell_height), cell_width, cell_height, fill=False, edgecolor='black', linewidth=1.0
            )
            ax.add_patch(rect_outline)

            ax.text(x + cell_width / 2, y - cell_height / 2 + 0.18 * scale,
                    f'M{machine}', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')

            ax.text(x + cell_width / 2, y - cell_height / 2 - 0.13 * scale,
                    f'{time}t', ha='center', va='center', fontsize=10, color='black')

    ax.set_xlim(-0.8, n_machines * (cell_width + spacing_x))
    ax.set_ylim(-0.6, title_y + 0.4)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight',
                facecolor='#F0F0F0', edgecolor='none')
    plt.close()
    return
