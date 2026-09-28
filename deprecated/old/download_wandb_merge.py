import os
import argparse
import wandb
import pandas as pd

from utils import sort_df, \
    DATASETS_UFGIR, METHODS_DIC, METHODS_RESNET, METHODS_VIT

CONFIG_COLS = [
    'serial', 'dataset_name', 'model_name', 'freeze_backbone',
    'lr', 'base_lr', 'seed', 'epochs', 'image_size', 'batch_size',
    'num_images_train', 'num_images_val',
]

SUMMARY_COLS = [
    'val_acc_species', 'val_acc_family', 'val_acc_order', 'val_loss', 'ap_w'
    'train_acc_species', 'train_acc_family', 'train_acc_order', 'train_loss',
    'time_total', 'max_memory', 'flops',
    'no_params', 'no_params_trainable', 'throughput', 
    'cka_avg_test', 'cka_avg_train', 'dist_avg_train', 'dist_avg_test', 'dist_norm_avg_train', 'dist_norm_avg_test', 'l2_norm_avg_train', 'l2_norm_avg_test',
    'cka_0_train', 'cka_0_test', 'cka_11_train', 'cka_11_test', 'cka_15_train', 'cka_15_test',
    'cka_high_mean_train', 'cka_mid_mean_train', 'cka_low_mean_train',
    'cka_high_mean_test', 'cka_mid_mean_test', 'cka_low_mean_test',
    'MSC_train', 'MSC_val', 'V_intra_train', 'V_intra_val', 'S_inter_train', 'S_inter_val',
    'clustering_diversity_train', 'clustering_diversity_val', 'spectral_diversity_train', 'spectral_diversity_val'
]

SORT_COLS = [
    'dataset_name', 'serial', 'model_name',
    'lr', 'seed', 'host', 'batch_size',
]

def get_wandb_project_runs(project, serials=None):
    api = wandb.Api()

    if serials:
        runs = api.runs(path=project, per_page=2000,
                        filters={'$or': [{'config.serial': s} for s in serials]})
    else:
        runs = api.runs(path=project, per_page=2000)

    print('Downloaded runs: ', len(runs))
    return runs

def make_df(runs, config_cols, summary_cols):
    data_list_dics = []

    for i, run in enumerate(runs):
        run_data = {}
        try:
            host = {'host': run.metadata.get('host')}
        except:
            print(run)
            host = {'host': None}
        cfg = {col: run.config.get(col, None) for col in config_cols}
        summary = {col: run.summary.get(col, None) for col in summary_cols}

        run_data.update(host)
        run_data.update(cfg)
        run_data.update(summary)

        data_list_dics.append(run_data)

        if (i + 1) % 10 == 0:
            print(f'{i}/{len(runs)}')

    df = pd.DataFrame.from_dict(data_list_dics)
    print(df.head())
    return df

def merge_cka_stats(df_main, runs, serial_fz=25, serial_ft=27):
    metrics = [
        'cka_avg_test', 'cka_avg_train', 'dist_avg_train', 'dist_avg_test',
        'dist_norm_avg_train', 'dist_norm_avg_test',
        'l2_norm_avg_train', 'l2_norm_avg_test',
        'cka_high_mean_train', 'cka_mid_mean_train', 'cka_low_mean_train',
        'cka_high_mean_test', 'cka_mid_mean_test', 'cka_low_mean_test',
        'cka_0_train', 'cka_0_test', 'cka_11_train', 'cka_11_test', 'cka_15_train', 'cka_15_test',
    ]

    def empty_df(suffix=""):
        columns = ['model_name', 'dataset_name'] + [m + suffix for m in metrics]
        return pd.DataFrame(columns=columns)

    # Filter frozen and fine-tuned runs
    runs_frozen = [run for run in runs if run.config.get('serial') == serial_fz]
    runs_ft = [run for run in runs if run.config.get('serial') == serial_ft]

    # Frozen — original names
    if runs_frozen:
        frozen_data = []
        for run in runs_frozen:
            entry = {
                'model_name': run.config.get('model_name'),
                'dataset_name': run.config.get('dataset_name'),
            }
            for m in metrics:
                entry[m] = run.summary.get(m)
            frozen_data.append(entry)
        df_frozen = pd.DataFrame(frozen_data)
    else:
        df_frozen = empty_df()

    # Fine-tuned — metrics with _ft suffix
    if runs_ft:
        ft_data = []
        for run in runs_ft:
            entry = {
                'model_name': run.config.get('model_name'),
                'dataset_name': run.config.get('dataset_name'),
            }
            for m in metrics:
                entry[m + '_ft'] = run.summary.get(m)
            ft_data.append(entry)
        df_ft = pd.DataFrame(ft_data)
    else:
        df_ft = empty_df('_ft')

    # Merge
    df_merged = df_main.merge(df_frozen, on=['model_name', 'dataset_name'], how='left')
    df_merged = df_merged.merge(df_ft, on=['model_name', 'dataset_name'], how='left')

    # Clean up any _x/_y suffixes
    for col in df_merged.columns:
        if col.endswith('_x'):
            base = col[:-2]
            alt = base + '_y'
            if alt in df_merged.columns:
                df_merged[base] = df_merged[col].combine_first(df_merged[alt])
                df_merged.drop([col, alt], axis=1, inplace=True)
            else:
                df_merged.rename(columns={col: base}, inplace=True)
        elif col.endswith('_y'):
            base = col[:-2]
            if base not in df_merged.columns:
                df_merged.rename(columns={col: base}, inplace=True)

    return df_merged



def merge_intrainter_stats(df_main, runs):
    metrics = [
        'MSC_train', 'MSC_val', 'V_intra_train', 'V_intra_val',
        'S_inter_train', 'S_inter_val',
        'clustering_diversity_train', 'clustering_diversity_val',
        'spectral_diversity_train', 'spectral_diversity_val'
    ]

    def empty_df(suffix=""):
        columns = ['model_name', 'dataset_name'] + [m + suffix for m in metrics]
        return pd.DataFrame(columns=columns)

    runs_frozen = [run for run in runs if run.config.get('serial') == 26]
    runs_ft = [run for run in runs if run.config.get('serial') == 28]

    if runs_frozen:
        frozen_data = []
        for run in runs_frozen:
            entry = {
                'model_name': run.config.get('model_name'),
                'dataset_name': run.config.get('dataset_name')
            }
            for m in metrics:
                entry[m] = run.summary.get(m)
            frozen_data.append(entry)
        df_frozen = pd.DataFrame(frozen_data)
    else:
        df_frozen = empty_df()

    if runs_ft:
        ft_data = []
        for run in runs_ft:
            entry = {
                'model_name': run.config.get('model_name'),
                'dataset_name': run.config.get('dataset_name')
            }
            for m in metrics:
                entry[m + '_ft'] = run.summary.get(m)
            ft_data.append(entry)
        df_ft = pd.DataFrame(ft_data)
    else:
        df_ft = empty_df('_ft')

    # Merge frozen and fine-tuned
    df_merged = df_main.merge(df_frozen, on=['model_name', 'dataset_name'], how='left')
    df_merged = df_merged.merge(df_ft, on=['model_name', 'dataset_name'], how='left')

    # Fix _x and _y suffixes if they exist
    for col in df_merged.columns:
        if col.endswith('_x'):
            base = col[:-2]
            alt = base + '_y'
            if alt in df_merged.columns:
                # Fill _x with _y if _x is null, then drop _y
                df_merged[base] = df_merged[col].combine_first(df_merged[alt])
                df_merged.drop([col, alt], axis=1, inplace=True)
            else:
                df_merged.rename(columns={col: base}, inplace=True)
        elif col.endswith('_y'):
            base = col[:-2]
            if base not in df_merged.columns:
                df_merged.rename(columns={col: base}, inplace=True)

    return df_merged



def load_and_process_pretraining_stats(stats_file, stats_resnet_file):
    # Load the two CSV files
    df_stats = pd.read_csv(stats_file)
    df_stats_resnet = pd.read_csv(stats_resnet_file)

    # Add "hi-" prefix to the 'pt' column in both DataFrames
    df_stats['pt'] = 'hi' + df_stats['pt'].astype(str)
    df_stats_resnet['pt'] = 'hi' + df_stats_resnet['pt'].astype(str)

    # Keep only the relevant columns: 'pt', 'augreg', 'acc_in1k'
    df_stats = df_stats[['pt', 'augreg', 'acc_in1k']]
    df_stats_resnet = df_stats_resnet[['pt', 'augreg', 'acc_in1k']]

    # Concatenate the two DataFrames
    df_combined = pd.concat([df_stats, df_stats_resnet], ignore_index=True)
    return df_combined

def merge_pretraining_stats(df, stats_df):
    # Merge the W&B DataFrame with the pretraining stats DataFrame
    # Match 'model_name' from df with 'pt' from stats_df
    df = df.merge(stats_df, how='left', left_on='model_name', right_on='pt')
    
    # Drop the redundant 'pt' column after merging
    if 'pt' in df.columns:
        df = df.drop(columns=['pt'])
    
    return df

def sort_save_df(df, fp, sort_cols=['serial']):
    df = df.sort_values(by=sort_cols, ascending=[True for _ in sort_cols])
    df.to_csv(fp, header=True, index=False)
    return 0

def parse_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--project_name', type=str, default='nycu_pcs/Hierarchical',
                        help='project_entity/project_name')
    # Filters
    parser.add_argument('--serials', nargs='+', type=int, default=[23, 24, 25, 26, 27, 28])
    parser.add_argument('--config_cols', nargs='+', type=str, default=CONFIG_COLS)
    parser.add_argument('--summary_cols', nargs='+', type=str, default=SUMMARY_COLS)
    # Output
    parser.add_argument('--output_file', default='hierarchical_test.csv', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str, default='data',
                        help='The directory where results will be stored')
    parser.add_argument('--sort_cols', nargs='+', type=str, default=SORT_COLS)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = os.path.join(args.results_dir, args.output_file)

    # Load and process the pretraining stats CSVs
    stats_df = load_and_process_pretraining_stats('data/stats_pretrainings.csv', 'data/stats_pretrainings_resnet.csv')

    # Get W&B runs and create the initial DataFrame
    runs = get_wandb_project_runs(args.project_name, args.serials)
    df = make_df(runs, args.config_cols, args.summary_cols)

    # Merge the pretraining stats into the DataFrame
    df = merge_pretraining_stats(df, stats_df)

    # Merge CKA stats from runs_cka (e.g., serial 25)
    df = merge_cka_stats(df, runs)

    df = merge_intrainter_stats(df, runs)

    # Sort and save the updated DataFrame
    sort_save_df(df, args.output_file, args.sort_cols)

    return 0

if __name__ == '__main__':
    main()