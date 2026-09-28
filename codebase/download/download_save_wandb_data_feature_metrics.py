import os
import argparse
import wandb
import pandas as pd


CONFIG_COLS = [
    'serial', 'dataset_name', 'model_name',
]

SUMMARY_COLS_GROUP1 = [
    'cka_avg_test', 'cka_avg_train', 'dist_avg_train', 'dist_avg_test',
    'dist_norm_avg_train', 'dist_norm_avg_test', 'l2_norm_avg_train', 'l2_norm_avg_test',
    'dist_0_train', 'dist_0_test', 'dist_11_train', 'dist_11_test', 'dist_15_train', 'dist_15_test',
    'cka_0_train', 'cka_0_test', 'cka_11_train', 'cka_11_test', 'cka_15_train', 'cka_15_test',
    'cka_high_mean_train', 'cka_mid_mean_train', 'cka_low_mean_train',
    'cka_high_mean_test', 'cka_mid_mean_test', 'cka_low_mean_test', 
    'dist_intra_0_train', 'dist_intra_0_test', 'dist_inter_0_train', 'dist_inter_0_test',
    'dist_intra_11_train', 'dist_intra_11_test', 'dist_inter_11_train', 'dist_inter_11_test',
    'dist_intra_15_train', 'dist_intra_15_test', 'dist_inter_15_train', 'dist_inter_15_test',
    'dist_intra_avg_train', 'dist_intra_avg_test', 'dist_inter_avg_train', 'dist_inter_avg_test',
]
SUMMARY_COLS_GROUP2 = [
    'MSC_train', 'MSC_test', 'V_intra_train', 'V_intra_test', 'S_inter_train', 'S_inter_test',
    'clustering_diversity_train', 'clustering_diversity_test',
    'spectral_diversity_train', 'spectral_diversity_test',
]
SUMMARY_COLS = SUMMARY_COLS_GROUP1 + SUMMARY_COLS_GROUP2

SORT_COLS = [
    'dataset_name', 'model_name',
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
    print(len(df), df.columns, df.iloc[0])

    return df


def add_ft_fz_suffixes(df, serial_fz=25, serial_ft=27, cols=SUMMARY_COLS_GROUP1):
    df = df[CONFIG_COLS + cols].copy(deep=False)

    # Filter frozen and fine-tuned runs
    df_fz = df[df['serial'] == serial_fz].copy(deep=False)
    df_ft = df[df['serial'] == serial_ft].copy(deep=False)

    # renamed_metrics_fz = {m: f'{m}_fz' for m in cols}
    renamed_metrics_ft = {m: f'{m}_ft' for m in cols}

    # df_fz = df_fz.rename(columns=renamed_metrics_fz)
    df_ft = df_ft.rename(columns=renamed_metrics_ft)

    # print(len(df_fz), df_fz.columns, df_fz.iloc[0])
    # print(len(df_ft), df_ft.columns, df_ft.iloc[0])

    df_fz = df_fz.drop(columns=['serial'])
    df_ft = df_ft.drop(columns=['serial'])
    df_merged = pd.merge(df_fz, df_ft, how='left', on=['dataset_name', 'model_name'])

    print(len(df_merged), df_merged.columns, df_merged.iloc[0])

    return df_merged


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
    parser.add_argument('--serials', nargs='+', type=int, default=[25, 26, 27, 28])
    parser.add_argument('--config_cols', nargs='+', type=str, default=CONFIG_COLS)
    parser.add_argument('--summary_cols', nargs='+', type=str, default=SUMMARY_COLS)

    # Output
    parser.add_argument('--output_file', default='hierarchical_feature_metrics.csv', type=str,
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

    # Get W&B runs and create the initial DataFrame
    runs = get_wandb_project_runs(args.project_name, args.serials)
    df = make_df(runs, args.config_cols, args.summary_cols)

    # add suffixes for ft and fz metrics
    # df_merged_group1 = add_ft_fz_suffixes(df, serial_fz=25, serial_ft=27, cols=SUMMARY_COLS_GROUP1)
    # df_merged_group1 = add_ft_fz_suffixes(df, serial_fz=35, serial_ft=36, cols=SUMMARY_COLS_GROUP1)
    df_merged_group2 = add_ft_fz_suffixes(df, serial_fz=26, serial_ft=28, cols=SUMMARY_COLS_GROUP2)

    # merge into a single dataframe
    # df = pd.merge(df_merged_group1, df_merged_group2, how='left', on=['dataset_name', 'model_name'])
    df = df_merged_group2
    print(df)

    # Sort and save the updated DataFrame
    sort_save_df(df, args.output_file, args.sort_cols)

    return 0

if __name__ == '__main__':
    main()