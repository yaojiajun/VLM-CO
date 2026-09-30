def draw_v2(instance: dict, out_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    processing_times = instance['processing_times']  # [machines × jobs]

    COLORS = [
        '#A0C4FF',  # Soft blue
        '#FF6F61',  # Coral
        '#FFD166',  # Bright yellow
        '#06D6A0',  # Vibrant green
        '#EF476F',  # Strong red
        '#00BFFF',  # Sky blue
        '#FFD700',  # Golden yellow
        '#6A0572',  # Deep purple
        '#FFABAB',  # Soft red
    ]

    cell_width = 1.0
    cell_height = 0.9
    spacing_x = 0.25
    spacing_y = 0.15

    fig_width = max(9, 0.9 + n_jobs * (cell_width + spacing_x))
    fig_height = 1.5 + n_machines * (cell_height + spacing_y)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    title_y = n_machines * (cell_height + spacing_y) + 1.2
    ax.text(n_jobs * (cell_width + spacing_x) / 2, title_y + 0.3,
            f'Permutation Flow Shop Scheduling Problem: {n_jobs} Jobs × {n_machines} Machines',
            ha='center', va='top', fontsize=16, fontweight='bold', color='#2E2E2E')
    ax.text(n_jobs * (cell_width + spacing_x) / 2, title_y,
            'Processing times (t) for each Job on each Machine, color coded by Machine',
            ha='center', va='top', fontsize=10, style='italic', color='#777777')

    # Column headers: Job0, Job1, ...
    for job_idx in range(n_jobs):
        x = job_idx * (cell_width + spacing_x) + cell_width / 2
        y = title_y - 0.7
        ax.text(x, y, f'Job{job_idx + 1}', ha='center', va='center',
                fontsize=14, fontweight='bold', color='#2c2c2c')

    # Rows: each row is a machine
    for machine_idx in range(n_machines):
        y = title_y - 1.5 - machine_idx * (cell_height + spacing_y)

        # Row label
        ax.text(-0.6, y - cell_height / 2, f'M{machine_idx + 1}',
                ha='right', va='center', fontsize=12, fontweight='bold', color='#4A4A4A')

        # Cells for each job
        for job_idx in range(n_jobs):
            x = job_idx * (cell_width + spacing_x)
            proc_time = processing_times[machine_idx][job_idx]

            # Adjust color opacity based on processing time
            color = COLORS[machine_idx % len(COLORS)]
            rect = patches.Rectangle((x, y - cell_height), cell_width, cell_height,
                                     linewidth=2, edgecolor='black',
                                     facecolor=color, alpha=0.9)  # Increased opacity
            ax.add_patch(rect)

            # Improved visibility for processing time
            ax.text(x + cell_width / 2, y - cell_height / 2,
                    f't={proc_time}',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='black')  # Changed to black

    ax.set_xlim(-0.8, n_jobs * (cell_width + spacing_x))
    ax.set_ylim(-0.6, title_y + 0.4)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#FFFFFF', edgecolor='none')
    plt.close()
    return
