import os
import argparse
import numpy as np
import pandas as pd
from codebase.analyses.compute_correlations import compute_correlations

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('results_all', 'cost', 'cost_val_acc_level1.csv'),
                        help='filename for input .csv file')
    parser.add_argument('--acc_level', nargs='*', default=[],
                        help='Pick the acc_level')
    parser.add_argument('--output_folder', default=os.path.join('results_all', 'corr'), type=str,
                        help='File path')
    parser.add_argument('--y_axis', nargs='+', type=str, default=['acc_max', 'ada_ratio'],
                        choices=['acc_max', 'ada_ratio'],
                        help='Y-axis metrics for correlation (acc_max, ada_ratio, or both)')
    parser.add_argument('--train_only', action='store_true', default=False,
                        help='Use only train metrics if True')
    args = parser.parse_args()
    return args

def get_corr(df, x_metric, y_metric):
    # Compute correlations
    spearman_corr, pearson_corr, r_squared = compute_correlations(df, x_metric, y_metric)
    return spearman_corr, pearson_corr, r_squared

def main():
    args = parse_args()

    # Define metrics lists
    all_metrics = [
        'acc_in1k', 'MSC_train', 'MSC_test', 'V_intra_train', 'V_intra_test', 'S_inter_train', 'S_inter_test',
        'cis_clustering_diversity_train', 'cis_clustering_diversity_test', 'cis_spectral_diversity_train', 'cis_spectral_diversity_test',
        'cka_avg_train', 'cka_avg_test', 'dist_avg_train', 'dist_avg_test', 'dist_norm_avg_train',
        'dist_norm_avg_test', 'l2_norm_avg_train', 'l2_norm_avg_test',
        'cka_0_train', 'cka_0_test', 'cka_high_mean_train', 'cka_mid_mean_train', 'cka_low_mean_train',
        'cka_high_mean_test', 'cka_mid_mean_test', 'cka_low_mean_test', 'clustering_diversity_train', 'spectral_diversity_train',
        'cka_last_layer_train', 'cka_last_layer_test',
        'cis_cka_0_train', 'cis_cka_0_test', 'cis_cka_last_train', 'cis_cka_last_test',
        'cis_dist_0_train', 'cis_dist_0_test', 'cis_dist_last_train', 'cis_dist_last_test'
    ]

    train_metrics = [
        'acc_in1k', 'MSC_train', 'V_intra_train', 'S_inter_train',
        'cis_clustering_diversity_train', 'cis_spectral_diversity_train',
        'cka_avg_train', 'dist_avg_train', 'dist_norm_avg_train',
        'l2_norm_avg_train', 'cka_0_train', 'cka_high_mean_train',
        'cka_mid_mean_train', 'cka_low_mean_train', 'clustering_diversity_train',
        'spectral_diversity_train', 'cka_last_layer_train',
        'cis_cka_0_train', 'cis_cka_last_train', 'cis_dist_0_train',
        'cis_dist_last_train', 
        'cis_cka_high_mean_train', 
        'cis_cka_low_mean_train',
        'cka_inv_low_mean_train', 'cka_inv_high_mean_train',
        'cis_cka_inv_low_mean_train', 'cis_cka_inv_high_mean_train'
    ]

    # Select metrics based on train_only flag
    metrics = train_metrics if args.train_only else all_metrics

    # Read input CSV
    df = pd.read_csv(args.input_file)

    # Ensure serial is treated as a string to avoid type mismatches
    df['serial'] = df['serial'].astype(str)

    # Add model_group column
    df['model_group'] = df['method'].apply(
        lambda x: 'vit' if 'hivit' in x.lower() or 'hideit' in x.lower() else 'resnet' if 'hiresnet' in x.lower() else 'unknown'
    )

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Get unique datasets and serials
    datasets = df['dataset_name'].unique()
    serials = df['serial'].unique()
    print(f"Unique datasets: {datasets}")
    print(f"Unique serials: {serials}")

    # Process each y_axis metric
    for y_metric in args.y_axis:
        results = []
        for dataset in datasets:
            for serial in serials:
                # Individual ViT and ResNet correlations
                for model_group in ['vit', 'resnet']:
                    # Add _ft suffix for serial 23
                    metric_suffix = '_ft' if serial == '23' else ''
                    for x_metric in metrics:
                        if x_metric == 'acc_in1k':
                            x_metric_adjusted = 'acc_in1k'
                        else:
                            x_metric_adjusted = f"{x_metric}{metric_suffix}"
                        # Extract split (train/test) from metric name
                        split = 'train' if x_metric_adjusted.endswith('_train') or x_metric_adjusted.endswith('_train_ft') else 'test' if x_metric_adjusted.endswith('_test') else 'none'
                        # Clean metric name for output (remove _train, _test, or _train_ft)
                        clean_metric = x_metric_adjusted.replace('_train_ft', '').replace('_train', '').replace('_test', '')
                        
                        # Adjust cka_last_layer for each model group
                        if x_metric in ['cka_last_layer_train', 'cka_last_layer_test']:
                            if model_group == 'resnet':
                                x_metric_adjusted = f"cka_15_train{metric_suffix}" if x_metric == 'cka_last_layer_train' else 'cka_15_test'
                            elif model_group == 'vit':
                                x_metric_adjusted = f"cka_11_train{metric_suffix}" if x_metric == 'cka_last_layer_train' else 'cka_11_test'
                        
                        # Filter data for the current dataset, serial, and model_group
                        group_df = df[(df['dataset_name'] == dataset) & (df['serial'] == serial) & (df['model_group'] == model_group)]
                        
                        # Debugging output
                        print(f"\nFiltering for dataset: {dataset}, serial: {serial}, model_group: {model_group}")
                        print(f"Using x_metric_adjusted: {x_metric_adjusted}")
                        print(f"Number of rows in group_df: {len(group_df)}")
                        if not group_df.empty:
                            print(f"Values for {y_metric}: {group_df[y_metric].values}")
                            print(f"Values for {x_metric_adjusted}: {group_df[x_metric_adjusted].values}")
                            print(f"Serial values in group_df: {group_df['serial'].values}")
                            print(f"Model group values in group_df: {group_df['model_group'].values}")
                            
                            # Compute correlations for the model group
                            spearman, pearson, r_squared = get_corr(group_df, x_metric_adjusted, y_metric)
                            
                            # Store result
                            results.append({
                                'serial': serial,
                                'dataset_name': dataset,
                                'model_group': model_group,
                                'metric': clean_metric,
                                'split': split,
                                'spearman': spearman,
                                'pearson': pearson,
                                'r_squared': r_squared,
                                'type': y_metric
                            })
                        else:
                            print(f"No data found for dataset: {dataset}, serial: {serial}, model_group: {model_group}")

                # ViT+RN+FT and ViT+RN+FZ correlations
                for model_group in ['both']:
                    metric_suffix = '_ft' if serial == '23' else ''
                    for x_metric in metrics:
                        if x_metric == 'acc_in1k':
                            x_metric_adjusted = 'acc_in1k'
                        else:
                            x_metric_adjusted = f"{x_metric}{metric_suffix}"
                        # Extract split (train/test) from metric name
                        split = 'train' if x_metric_adjusted.endswith('_train') or x_metric_adjusted.endswith('_train_ft') else 'test' if x_metric_adjusted.endswith('_test') else 'none'
                        clean_metric = x_metric_adjusted.replace('_train_ft', '').replace('_train', '').replace('_test', '')
                        
                        # Adjust cka_last_layer for combined ViT+RN
                        if x_metric in ['cka_last_layer_train', 'cka_last_layer_test']:
                            x_metric_adjusted = f"cka_last_{'train' if x_metric == 'cka_last_layer_train' else 'test'}{metric_suffix}"
                        
                        # Filter data for both ViT and ResNet
                        group_df = df[(df['dataset_name'] == dataset) & (df['serial'] == serial) & (df['model_group'].isin(['vit', 'resnet']))]
                        
                        # Debugging output
                        print(f"\nFiltering for dataset: {dataset}, serial: {serial}, model_group: {model_group}")
                        print(f"Using x_metric_adjusted: {x_metric_adjusted}")
                        print(f"Number of rows in group_df: {len(group_df)}")
                        if not group_df.empty:
                            print(f"Values for {y_metric}: {group_df[y_metric].values}")
                            print(f"Values for {x_metric_adjusted}: {group_df[x_metric_adjusted].values}")
                            print(f"Serial values in group_df: {group_df['serial'].values}")
                            print(f"Model group values in group_df: {group_df['model_group'].values}")
                            
                            # Compute correlations
                            spearman, pearson, r_squared = get_corr(group_df, x_metric_adjusted, y_metric)
                            
                            # Store result
                            results.append({
                                'serial': serial,
                                'dataset_name': dataset,
                                'model_group': model_group,
                                'metric': clean_metric,
                                'split': split,
                                'spearman': spearman,
                                'pearson': pearson,
                                'r_squared': r_squared,
                                'type': y_metric
                            })
                        else:
                            print(f"No data found for dataset: {dataset}, serial: {serial}, model_group: {model_group}")

                # y_var_fz vs metric_ft and y_var_ft vs metric_fz
                for model_group in ['accfzvsmetricft', 'accftvsmetricfz']:
                    if model_group == 'accfzvsmetricft' and serial == '24':
                        y_metric_adjusted = y_metric  # Frozen y_var
                        x_metric_suffix = '_ft'  # Fine-tuned x_metric
                    elif model_group == 'accftvsmetricfz' and serial == '23':
                        y_metric_adjusted = y_metric  # Fine-tuned y_var
                        x_metric_suffix = ''  # Frozen x_metric
                    else:
                        continue  # Skip irrelevant combinations
                    
                    for x_metric in metrics:
                        if x_metric == 'acc_in1k':
                            x_metric_adjusted = 'acc_in1k'
                        else:
                            x_metric_adjusted = f"{x_metric}{metric_suffix}"
                        # Extract split (train/test) from metric name
                        split = 'train' if x_metric_adjusted.endswith('_train') or x_metric_adjusted.endswith('_train_ft') else 'test' if x_metric_adjusted.endswith('_test') else 'none'
                        clean_metric = x_metric_adjusted.replace('_train_ft', '').replace('_train', '').replace('_test', '')
                        
                        # Adjust cka_last_layer for combined ViT+RN
                        if x_metric in ['cka_last_layer_train', 'cka_last_layer_test']:
                            x_metric_adjusted = f"cka_last_{'train' if x_metric == 'cka_last_layer_train' else 'test'}{x_metric_suffix}"
                        
                        # Filter data for both ViT and ResNet
                        group_df = df[(df['dataset_name'] == dataset) & (df['serial'] == serial) & (df['model_group'].isin(['vit', 'resnet']))]
                        
                        # Debugging output
                        print(f"\nFiltering for dataset: {dataset}, serial: {serial}, model_group: {model_group}")
                        print(f"Using y_metric_adjusted: {y_metric_adjusted}, x_metric_adjusted: {x_metric_adjusted}")
                        print(f"Number of rows in group_df: {len(group_df)}")
                        if not group_df.empty:
                            print(f"Values for {y_metric_adjusted}: {group_df[y_metric_adjusted].values}")
                            print(f"Values for {x_metric_adjusted}: {group_df[x_metric_adjusted].values}")
                            print(f"Serial values in group_df: {group_df['serial'].values}")
                            print(f"Model group values in group_df: {group_df['model_group'].values}")
                            
                            # Compute correlations
                            spearman, pearson, r_squared = get_corr(group_df, x_metric_adjusted, y_metric_adjusted)
                            
                            # Store result
                            results.append({
                                'serial': serial,
                                'dataset_name': dataset,
                                'model_group': model_group,
                                'metric': clean_metric,
                                'split': split,
                                'spearman': spearman,
                                'pearson': pearson,
                                'r_squared': r_squared,
                                'type': y_metric_adjusted
                            })
                        else:
                            print(f"No data found for dataset: {dataset}, serial: {serial}, model_group: {model_group}")

        # Create DataFrame from results
        result_df = pd.DataFrame(results)

        # Save overall CSV for this y_metric
        output_file = os.path.join(args.output_folder, f'correlations_{y_metric}.csv')
        result_df.to_csv(output_file, index=False)
        print(f"Saved correlations for {y_metric} to {output_file}")

        # Create per-metric CSVs
        for metric in result_df['metric'].unique():
            metric_df = result_df[result_df['metric'] == metric]
            if not metric_df.empty:
                output_file = os.path.join(args.output_folder, f'corr_{y_metric}_{metric}.csv')
                metric_df.to_csv(output_file, index=False)
                print(f"Saved correlations for {y_metric}, metric {metric} to {output_file}")

        # Create per-y_axis CSV (all metrics for this y_metric)
        if not result_df.empty:
            output_file = os.path.join(args.output_folder, f'corr_{y_metric}_all.csv')
            result_df.to_csv(output_file, index=False)
            print(f"Saved correlations for {y_metric} (all metrics) to {output_file}")

if __name__ == "__main__":
    main()