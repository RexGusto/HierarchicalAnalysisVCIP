import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse
from math import ceil

# Configuration dictionary for metric names
METRIC_NAME_MAP = {
    'acc_in1k': "ImageNet-1K Accuracy",
    'ada_ratio': "ADA Ratio",
    'cka_avg_train': "CKA (Train)",
    'cka_avg_test': "CKA (Test)",
    'dist_avg_train': "Distance (Train)",
    'dist_avg_test': "Distance (Test)",
    'dist_norm_avg_train': "Distance (Normalized, Train)",
    'dist_norm_avg_test': "Distance (Normalized, Test)",
    'l2_norm_avg_train': "L2 Norm (Train)",
    'l2_norm_avg_test': "L2 Norm (Test)",
    'cka_0_train': "CKA (First Layer, Train)",
    'cka_0_test': "CKA (First Layer, Test)",
    'cka_last_layer_train': "CKA (Last Layer, Train)",
    'cka_last_layer_test': "CKA (Last Layer, Test)",
    'cka_high_mean_train': "CKA (High Layers, Train)",
    'cka_high_mean_test': "CKA (High Layers, Test)",
    'cka_mid_mean_train': "CKA (Mid Layers, Train)",
    'cka_mid_mean_test': "CKA (Mid Layers, Test)",
    'cka_low_mean_train': "CKA (Low Layers, Train)",
    'cka_low_mean_test': "CKA (Low Layers, Test)",
    'msc_train': "MSC (Train)",
    'msc_test': "MSC (Test)",
    'v_intra_train': "Intra-Class Variance (Train)",
    'v_intra_test': "Intra-Class Variance (Test)",
    's_inter_train': "Inter-Class Separation (Train)",
    's_inter_test': "Inter-Class Separation (Test)",
    'cis_clustering_diversity_train': "CIS (Clustering, Train)",
    'cis_clustering_diversity_test': "CIS (Clustering, Test)",
    'cis_spectral_diversity_train': "CIS (Spectral, Train)",
    'cis_spectral_diversity_test': "CIS (Spectral, Test)",
    'clustering_diversity_train': "Clustering Diversity (Train)",
    'clustering_diversity_test': "Clustering Diversity (Test)",
    'spectral_diversity_train': "Spectral Diversity (Train)",
    'spectral_diversity_test': "Spectral Diversity (Test)",
    'cis_cka_0_train': "CIS (CKA First Layer, Train)",
    'cis_cka_0_test': "CIS (CKA First Layer, Test)",
    'cis_cka_last_train': "CIS (CKA Last Layer, Train)",
    'cis_cka_last_test': "CIS (CKA Last Layer, Test)",
    'cis_dist_0_train': "CIS (Dist. First Layer, Train)",
    'cis_dist_0_test': "CIS (Dist. First Layer, Test)",
    'cis_dist_last_train': "CIS (Dist. Last Layer, Train)",
    'cis_dist_last_test': "CIS (Dist Last Layer, Test)",
    'cis_cka_high_mean_train': "CIS (CKA High Layer)",
    'cis_cka_low_mean_train': "CIS (CKA Low Layer)",
    'cis_cka_inv_low_mean_train': "CIS (CKA Inv. Low Layer)",
    'cis_cka_inv_high_mean_train': "CIS (CKA Inv. High Layer)",
    'dist_intra_0_train': "Intra-class Dist. First Layer",
    'dist_inter_0_train': "Inter-class Dist. First Layer",
    'dist_intra_last_layer_train': "Intra-class Dist. Last Layer",
    'dist_inter_last_layer_train': "Inter-class Dist. Last Layer",
    'dist_intra_avg_train': "Intra-class Dist. Average of Layer",
    'dist_inter_avg_train': "Inter-class Dist. Average of Layer",
}

# Configuration dictionary for prefix titles
PREFIX_TITLE_MAP = {
    'ada_ratio': "Correlations for ADA Ratio",
    'acc_max': "Correlations for"
}

# Configuration dictionary for settings titles
SETTINGS_TITLE_MAP = {
    'bothaccftvsmetricfz': "FT Acc Vs FZ metric",
    'bothft': "RN+VIT FT",
    'bothft+fz': "RN+VIT FT+FZ",
    'bothfz': "RN+VIT FZ",
    'rnft': "RN FT",
    'rnfz': "RN FZ",
    'vitft': "VIT FT",
    'vitfz': "VIT FZ"
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate multi-plot image grid from a folder.')
    parser.add_argument('--input_folder', default='to_be_merged2', help='Input folder containing .png images')
    parser.add_argument('--output_folder', default='multi_plot_outputs', help='Result directory for generated plot')
    parser.add_argument('--output_file', default='multiplot', help='Name of the output file')
    parser.add_argument('--title', default='', help='Title extender for plot')
    parser.add_argument('--img_width', type=float, default=4, help='Width of each tiny plot in the grid')
    parser.add_argument('--img_height', type=float, default=3, help='Height of each tiny plot in the grid')
    parser.add_argument('--x_filter', nargs='*', default=[],
                        help='Only use the specified identifiers from x_axis. If left empty, use all the identifiers')
    parser.add_argument('--y_filter', nargs='*', default=[],
                        help='Only use the specified identifiers from y_axis. If left empty, use all the identifiers')
    parser.add_argument('--x_axis', choices=['dataset', 'setting', 'metric'], default='setting',
                        help='What to plot on x-axis: "dataset", "setting", or "metric"')
    parser.add_argument('--y_axis', choices=['dataset', 'setting', 'metric'], default='dataset',
                        help='What to plot on y-axis: "dataset", "setting", or "metric"')
    args = parser.parse_args()
    
    # Validate that x_axis and y_axis are different
    if args.x_axis == args.y_axis:
        parser.error("x_axis and y_axis must be different (cannot both be 'dataset', 'setting', or 'metric')")
    return args

def extract_sort_components(filename):
    base = os.path.basename(filename).replace(".png", "")
    parts = base.split("_")
    try:
        setting = parts[0]
        dataset = parts[-1]
        # Extract metric parts and check for acc_max or ada_ratio prefix
        metric_parts = parts[1:-1]
        # Check if first two elements form acc_max or ada_ratio
        prefix = None
        if len(metric_parts) >= 2:
            potential_prefix = f"{metric_parts[0]}_{metric_parts[1]}"
            if potential_prefix in ['acc_max', 'ada_ratio']:
                prefix = potential_prefix
                metric = "_".join(metric_parts[2:]) if len(metric_parts) > 2 else ""
            else:
                metric = "_".join(metric_parts)
        else:
            metric = "_".join(metric_parts)
        return dataset, setting, metric, prefix
    except IndexError:
        return "zzz", "zzz", "zzz", None

def get_sort_key(filename, args, row_priority, col_priority, y_values, x_values):
    dataset, setting, metric, _ = extract_sort_components(filename)
    
    # Assign row index based on axis configuration
    if args.y_axis == 'dataset':
        if args.y_filter:
            row_idx = args.y_filter.index(dataset) if dataset in args.y_filter else 99
        else:
            row_idx = y_values.index(dataset) if dataset in y_values else 99
    elif args.y_axis == 'setting':
        row_idx = row_priority.get(setting, 99) if row_priority else (y_values.index(setting) if setting in y_values else 99)
    else: 
        row_idx = row_priority.get(metric, 99) if row_priority else (y_values.index(metric) if metric in y_values else 99)
    
    # Assign column index based on axis configuration
    if args.x_axis == 'dataset':
        if args.x_filter:
            col_idx = args.x_filter.index(dataset) if dataset in args.x_filter else 99
        else:
            col_idx = x_values.index(dataset) if dataset in x_values else 99
    elif args.x_axis == 'setting':
        col_idx = col_priority.get(setting, 99) if col_priority else (x_values.index(setting) if setting in x_values else 99)
    else:
        col_idx = col_priority.get(metric, 99) if col_priority else (x_values.index(metric) if metric in x_values else 99)
    
    return (row_idx, col_idx)

def create_image_grid(input_folder, output_file, args, row_priority, col_priority, filter_component=None, filter_value=None, prefix=None):
    # Collect image files, optionally filtering by component
    image_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(".png")
    ]
    if filter_component and filter_value:
        image_files = [
            f for f in image_files
            if extract_sort_components(f)[
                0 if filter_component == 'dataset' else 1 if filter_component == 'setting' else 2
            ] == filter_value
        ]
    if prefix:
        image_files = [
            f for f in image_files
            if extract_sort_components(f)[3] == prefix
        ]
    if not image_files:
        return False

    # Determine unique values for y_axis and x_axis to set grid dimensions and labels
    y_values = []
    x_values = []
    for img_path in image_files:
        dataset, setting, metric, _ = extract_sort_components(img_path)
        y_val = dataset if args.y_axis == 'dataset' else setting if args.y_axis == 'setting' else metric
        x_val = dataset if args.x_axis == 'dataset' else setting if args.x_axis == 'setting' else metric
        if y_val not in y_values:
            y_values.append(y_val)
        if x_val not in x_values:
            x_values.append(x_val)
    
    # Filter y_values and x_values based on filters
    if args.y_filter:
        y_values = [v for v in args.y_filter if v in y_values]
    else:
        y_values = sorted(y_values)
    if args.x_filter:
        x_values = [v for v in args.x_filter if v in x_values]
    else:
        x_values = sorted(x_values)
    
    # Set grid dimensions using filter lengths when filters are provided
    grid_rows = len(args.y_filter) if args.y_filter else len(y_values)
    grid_cols = len(args.x_filter) if args.x_filter else len(x_values)
    
    # Ensure at least 1 row and column
    grid_rows = max(1, grid_rows)
    grid_cols = max(1, grid_cols)
    
    # Adjust figure height to accommodate title for large grids
    base_title_space = 0.3  # Base space for small grids
    additional_space = min(grid_rows * 0.015, 0.3)  # Cap the additional space at 0.3
    title_space = base_title_space + additional_space
    adjusted_height = grid_rows * args.img_height + title_space * 2  # Add space for title
        
    # Filter image files based on x_filter and y_filter
    filtered_image_files = []
    for img_path in image_files:
        dataset, setting, metric, _ = extract_sort_components(img_path)
        y_val = dataset if args.y_axis == 'dataset' else setting if args.y_axis == 'setting' else metric
        x_val = dataset if args.x_axis == 'dataset' else setting if args.x_axis == 'setting' else metric
        if (not args.y_filter or y_val in args.y_filter) and (not args.x_filter or x_val in args.x_filter):
            filtered_image_files.append(img_path)
    
    # Sort filtered image files
    filtered_image_files = sorted(filtered_image_files, key=lambda x: get_sort_key(x, args, row_priority, col_priority, y_values, x_values))
    
    # Determine the unused component for the title
    used_axes = {args.x_axis, args.y_axis}
    title_component = 'metric' if 'metric' not in used_axes else 'setting' if 'setting' not in used_axes else 'dataset'
    title_value = filter_value if filter_component == title_component else ''
    if title_component == 'metric' and title_value in METRIC_NAME_MAP:
        title_value = METRIC_NAME_MAP[title_value]
    elif title_component == 'dataset':
        title_value = title_value.upper() if title_value == 'cub' else title_value.capitalize()
    elif title_component == 'setting' and title_value in SETTINGS_TITLE_MAP:
        title_value = SETTINGS_TITLE_MAP[title_value]
    
    # Set title based on prefix and append args.title
    base_title = PREFIX_TITLE_MAP.get(prefix, "Correlations")
    title = f"{base_title} {args.title} vs {title_value}".strip()
    
    # Set up figure with adjusted height
    plt.rcParams["font.family"] = 'serif'
    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(grid_cols * args.img_width, adjusted_height),
        
    )
    axes = axes.flatten() if grid_rows * grid_cols > 1 else [axes]

    # Plot images from filtered files
    for i, (ax, img_path) in enumerate(zip(axes, filtered_image_files)):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('on')  # Turn axis on

        # Hide spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Hide ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])

        col_idx = i % grid_cols
        row_idx = i // grid_cols

        # Row label (y-axis)
        if col_idx == 0 and row_idx < grid_rows and row_idx < len(y_values):
            label_value = y_values[row_idx]
            if args.y_axis == 'dataset':
                label_text = label_value.upper() if label_value == 'cub' else label_value.capitalize()
            elif args.y_axis == 'metric':
                label_text = METRIC_NAME_MAP.get(label_value, label_value)
            elif args.y_axis == 'setting':
                label_text = SETTINGS_TITLE_MAP.get(label_value, label_value)
            ax.set_ylabel(label_text, fontsize=28)

        # Column label (x-axis)
        if row_idx == grid_rows - 1 and col_idx < grid_cols and col_idx < len(x_values):
            label_value = x_values[col_idx]
            if args.x_axis == 'dataset':
                label_text = label_value.upper() if label_value == 'cub' else label_value.capitalize()
            elif args.x_axis == 'metric':
                label_text = METRIC_NAME_MAP.get(label_value, label_value)
            elif args.x_axis == 'setting':
                label_text = SETTINGS_TITLE_MAP.get(label_value, label_value)
            ax.set_xlabel(label_text, fontsize=28)

    # Hide extra axes
    for ax in axes[len(filtered_image_files):]:
        ax.axis('off')

    # Set title and adjust top margin
    fig.suptitle(title, fontsize=28)
    fig.subplots_adjust(wspace=0.3)
    if grid_rows >= 5 and grid_cols >= 5:
        top_margin = 1.1 - (title_space / adjusted_height) 
        fig.subplots_adjust(top=top_margin)
    if grid_rows >= 8 and grid_cols >= 8:
        top_margin = 1.2 - (title_space / adjusted_height) 
        fig.subplots_adjust(top=top_margin)

    # Save plot
    plt.tight_layout(pad=0.5)
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    return True

def main():
    args = parse_arguments()
    
    # Collect all image files
    image_files = [f for f in os.listdir(args.input_folder) if f.endswith(".png")]
    
    # Collect unique datasets, settings, metrics, and prefixes
    datasets = set()
    settings = set()
    metrics = set()
    prefixes = set()
    for f in image_files:
        dataset, setting, metric, prefix = extract_sort_components(f)
        datasets.add(dataset)
        settings.add(setting)
        metrics.add(metric)
        if prefix:
            prefixes.add(prefix)
    
    # Print counts for debugging
    print(f"Unique datasets: {len(datasets)} ({sorted(datasets)})")
    print(f"Unique settings: {len(settings)} ({sorted(settings)})")
    print(f"Unique metrics: {len(metrics)} ({sorted(metrics)})")
    print(f"Unique prefixes: {len(prefixes)} ({sorted(prefixes)})")
    
    # Create priorities for row and column sorting
    row_priority = {item: i for i, item in enumerate(args.y_filter)} if args.y_filter else {}
    col_priority = {item: i for i, item in enumerate(args.x_filter)} if args.x_filter else {}
    
    # Determine the unused component for output filename
    used_axes = {args.x_axis, args.y_axis}
    name_component = 'metric' if 'metric' not in used_axes else 'setting' if 'setting' not in used_axes else 'dataset'
    
    # Get unique values for the name_component
    component_values = datasets if name_component == 'dataset' else settings if name_component == 'setting' else metrics
    
    # Create one plot per unique value of the name_component and prefix
    os.makedirs(args.output_folder, exist_ok=True)
    for value in sorted(component_values):
        for prefix in sorted(prefixes):
            output_file = os.path.join(args.output_folder, f"{args.output_file}_{prefix}_{value}.png")
            if create_image_grid(args.input_folder, output_file, args, row_priority, col_priority, 
                                filter_component=name_component, filter_value=value, prefix=prefix):
                print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()