import os
import argparse
import numpy as np
import pandas as pd

from utils import preprocess_df, add_all_cols_group, \
    round_combine_str_mean_std, sort_df, rename_var, \
    DATASETS_UFGIR, DATASETS_DIC, METHODS_DIC

agg_map = {
    'cka_avg_test': 'max',
    'cka_avg_train': 'max',
    'MSC_train': 'max', 'MSC_train_ft': 'max',
    'MSC_test': 'max', 'MSC_test_ft': 'max',
    'V_intra_train': 'max', 'V_intra_train_ft': 'max',
    'V_intra_test': 'max', 'V_intra_test_ft': 'max',
    'S_inter_train': 'max', 'S_inter_train_ft': 'max',
    'S_inter_test': 'max', 'S_inter_test_ft': 'max',
    'clustering_diversity_train': 'max', 'clustering_diversity_train_ft': 'max',
    'clustering_diversity_test': 'max', 'clustering_diversity_test_ft': 'max', 
    'spectral_diversity_train': 'max', 'spectral_diversity_train_ft': 'max',
    'spectral_diversity_test': 'max', 'spectral_diversity_test_ft': 'max',
    'dist_avg_train': 'max', 'dist_avg_test': 'max', 'dist_norm_avg_train': 'max', 'dist_norm_avg_test': 'max',
    'l2_norm_avg_train': 'max', 'l2_norm_avg_test': 'max',
    'cka_0_train': 'max', 'cka_0_test': 'max', 'cka_11_train': 'max', 'cka_11_test': 'max', 'cka_15_train': 'max', 'cka_15_test': 'max',
    'cka_avg_test_ft': 'max', 'cka_avg_train_ft': 'max', 'dist_avg_train_ft': 'max', 'dist_avg_test_ft': 'max', 'dist_norm_avg_train_ft': 'max', 
    'dist_norm_avg_test_ft': 'max',
    'l2_norm_avg_train_ft': 'max', 'l2_norm_avg_test_ft': 'max',
    'cka_0_train_ft': 'max', 'cka_0_test_ft': 'max', 'cka_11_train_ft': 'max', 'cka_11_test_ft': 'max', 'cka_15_train_ft': 'max', 'cka_15_test_ft': 'max',
    'cka_high_mean_train': 'max', 'cka_mid_mean_train': 'max', 'cka_low_mean_train': 'max',
    'cka_high_mean_test': 'max', 'cka_mid_mean_test': 'max', 'cka_low_mean_test': 'max',
    'cka_high_mean_train_ft': 'max', 'cka_mid_mean_train_ft': 'max', 'cka_low_mean_train_ft': 'max',
    'cka_high_mean_test_ft': 'max', 'cka_mid_mean_test_ft': 'max', 'cka_low_mean_test_ft': 'max',
}

def aggregate_results_main(df, fp=None, serials=None,
                           add_method_avg=True, add_dataset_avg=False):

    if add_method_avg:
        df = add_all_cols_group(df, 'dataset_name')
    if add_dataset_avg:
        df = add_all_cols_group(df, 'method')

    group_keys = ['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k']

    df_temp = df

    # Drop duplicate 'lr' columns, keeping the first
    df_temp = df_temp.loc[:, ~df.columns.duplicated()]

    # Sort to ensure the lowest lr appears first
    df_sorted = df_temp.sort_values(by='lr')

    # Drop duplicates to keep the row with lowest lr for each group
    df_last_acc = df_sorted.drop_duplicates(subset=group_keys, keep='first')
    df_last_acc = df_last_acc[group_keys + ['acc']].copy()
    df_last_acc.rename(columns={'acc': 'last_acc'}, inplace=True)


    df_extra = df.groupby(['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], as_index=False).agg(agg_map)
    df_std = df.groupby(['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], as_index=False).agg({'acc': 'std'})
    df_max = df.groupby(['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], as_index=False).agg({'acc': 'max'})
    df_min = df.groupby(['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], as_index=False).agg({'acc': 'min'})
    # print(df_cka)
    df = df.groupby(['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], as_index=False).agg({'acc': 'mean'})
    
    df['acc_std'] = df_std['acc']
    df['acc_max'] = df_max['acc']
    df['acc_min'] = df_min['acc']

    # Merge the extra metrics back in
    df = df.merge(df_last_acc, on=group_keys, how='left')

    # For finetuned checkpoints, it refers to the last learning rate = last accuracy
    # For frozen, use the max frozen accuracy as reference
    # df['acc_combined'] = df.apply(
    #     lambda row: row['last_acc'] if row['serial'] == 23 else row['acc_max'] if row['serial'] == 24 else np.nan,
    #     axis=1
    # )

    # For new plots as the finetuned checkpoins refers to the best settings 
    df['acc_combined'] = df.apply(
        lambda row: row['acc_max'] if row['serial'] == 23 else row['acc_max'] if row['serial'] == 24 else np.nan,
        axis=1
    )

    df = df.merge(df_extra, on=['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], how='left')

    # Compute CIS = diversity * acc_in1k
    df['cis_spectral_train'] = df['spectral_diversity_train'] * df['acc_in1k']
    df['cis_spectral_test'] = df['spectral_diversity_test'] * df['acc_in1k']
    df['cis_clustering_train'] = df['clustering_diversity_train'] * df['acc_in1k']
    df['cis_clustering_test'] = df['clustering_diversity_test'] * df['acc_in1k']

    df['cis_spectral_train_ft'] = df['spectral_diversity_train_ft'] * df['acc_in1k']
    df['cis_spectral_test_ft'] = df['spectral_diversity_test_ft'] * df['acc_in1k']
    df['cis_clustering_train_ft'] = df['clustering_diversity_train_ft'] * df['acc_in1k']
    df['cis_clustering_test_ft'] = df['clustering_diversity_test_ft'] * df['acc_in1k']

    # Combine cka very last layer
    df['cka_last_combined_train'] = np.nan
    df['cka_last_combined_test'] = np.nan
    df['cka_last_combined_train_ft'] = np.nan
    df['cka_last_combined_test_ft'] = np.nan
    
    for idx in df.index:
        method = df.at[idx, 'method']
        # ViT 12 layers (0-11)
        if 'hivit' in method or 'hideit' in method:
            df.at[idx, 'cka_last_combined_train'] = df.at[idx, 'cka_11_train']
            df.at[idx, 'cka_last_combined_test'] = df.at[idx, 'cka_11_test']
            df.at[idx, 'cka_last_combined_train_ft'] = df.at[idx, 'cka_11_train_ft']
            df.at[idx, 'cka_last_combined_test_ft'] = df.at[idx, 'cka_11_test_ft']
        # ResNet 16 layers (0-16)
        elif 'hiresnet50' in method:
            df.at[idx, 'cka_last_combined_train'] = df.at[idx, 'cka_15_train']
            df.at[idx, 'cka_last_combined_test'] = df.at[idx, 'cka_15_test']
            df.at[idx, 'cka_last_combined_train_ft'] = df.at[idx, 'cka_15_train_ft']
            df.at[idx, 'cka_last_combined_test_ft'] = df.at[idx, 'cka_15_test_ft']

    # Combine ft and fz values into column with _combined suffix
    for col in df.columns:
        if col.endswith('_ft'):
            base_col = col.replace('_ft', '')
            new_col = col.replace('_ft', '_combined')
            df[new_col] = df.apply(
                lambda row: row[base_col] if row['serial'] == 24 else row[col] if row['serial'] == 23 else np.nan,
                axis=1
            )
    
    df = sort_df(df)

    # Calculate logits
    df['acc_logit'] = 0.0
    df['acc_max_logit'] = 0.0
    df['acc_std_logit'] = 0.0

    epsilon = 1e-10
    for idx in df.index:
        # Get acc and acc_max for the current row
        acc = df.at[idx, 'acc'] /100
        acc_max = df.at[idx, 'acc_max'] /100
        acc_std = df.at[idx, 'acc_std'] /100

        clipped_acc = max(min(acc, 1 - epsilon), epsilon)
        clipped_acc_max = max(min(acc_max, 1 - epsilon), epsilon)
        clipped_acc_std = max(min(acc_std, 1 - epsilon), epsilon)

        # logit(p) = log(p / (1 - p))
        df.at[idx, 'acc_logit'] = np.log(clipped_acc / (1 - clipped_acc))
        df.at[idx, 'acc_max_logit'] = np.log(clipped_acc_max / (1 - clipped_acc_max))
        df.at[idx, 'acc_std_logit'] = np.log(clipped_acc_std / (1 - clipped_acc_std))

    df = round_combine_str_mean_std(df, col='acc')

    if fp:
        df.to_csv(fp, header=True, index=False)
    return df


def aggregate_results_main_2(df, fp=None, serials=None,
                            add_method_avg=True, add_dataset_avg=False):

    if add_method_avg:
        df = add_all_cols_group(df, 'dataset_name')
    if add_dataset_avg:
        df = add_all_cols_group(df, 'method')

    group_keys = ['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k']

    df_temp = df

    # Drop duplicate 'lr' columns, keeping the first
    df_temp = df_temp.loc[:, ~df.columns.duplicated()]

    # Sort to ensure the lowest lr appears first
    df_sorted = df_temp.sort_values(by='lr')

    # Drop duplicates to keep the row with lowest lr for each group
    df_last_acc = df_sorted.drop_duplicates(subset=group_keys, keep='first')
    df_last_acc = df_last_acc[group_keys + ['acc_2']].copy()
    df_last_acc.rename(columns={'acc_2': 'last_acc'}, inplace=True)

    df_extra = df.groupby(['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], as_index=False).agg(agg_map)
    df_std = df.groupby(['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], as_index=False).agg({'acc_2': 'std'})
    df_max = df.groupby(['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], as_index=False).agg({'acc_2': 'max'})
    df_min = df.groupby(['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], as_index=False).agg({'acc_2': 'min'})
    # print(df_cka)
    df = df.groupby(['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], as_index=False).agg({'acc_2': 'mean'})
    
    df['acc_2_std'] = df_std['acc_2']
    df['acc_2_max'] = df_max['acc_2']
    df['acc_2_min'] = df_min['acc_2']

    # Merge the extra metrics back in
    df = df.merge(df_last_acc, on=group_keys, how='left')

    # For finetuned checkpoints, it refers to the last learning rate = last accuracy
    # For frozen, use the max frozen accuracy as reference
    # df['acc_combined'] = df.apply(
    #     lambda row: row['last_acc'] if row['serial'] == 23 else row['acc_max'] if row['serial'] == 24 else np.nan,
    #     axis=1
    # )

    # For new plots as the finetuned checkpoins refers to the best settings 
    df['acc_2_combined'] = df.apply(
        lambda row: row['acc_2_max'] if row['serial'] == 23 else row['acc_2_max'] if row['serial'] == 24 else np.nan,
        axis=1
    )

    df = df.merge(df_extra, on=['serial', 'setting', 'dataset_name', 'method', 'augreg', 'acc_in1k'], how='left')

    # Compute CIS = diversity * acc_in1k
    df['cis_spectral_train'] = df['spectral_diversity_train'] * df['acc_in1k']
    df['cis_spectral_test'] = df['spectral_diversity_test'] * df['acc_in1k']
    df['cis_clustering_train'] = df['clustering_diversity_train'] * df['acc_in1k']
    df['cis_clustering_test'] = df['clustering_diversity_test'] * df['acc_in1k']

    df['cis_spectral_train_ft'] = df['spectral_diversity_train_ft'] * df['acc_in1k']
    df['cis_spectral_test_ft'] = df['spectral_diversity_test_ft'] * df['acc_in1k']
    df['cis_clustering_train_ft'] = df['clustering_diversity_train_ft'] * df['acc_in1k']
    df['cis_clustering_test_ft'] = df['clustering_diversity_test_ft'] * df['acc_in1k']

    # Combine cka very last layer
    df['cka_last_combined_train'] = np.nan
    df['cka_last_combined_test'] = np.nan
    df['cka_last_combined_train_ft'] = np.nan
    df['cka_last_combined_test_ft'] = np.nan
    
    for idx in df.index:
        method = df.at[idx, 'method']
        # ViT 12 layers (0-11)
        if 'hivit' in method or 'hideit' in method:
            df.at[idx, 'cka_last_combined_train'] = df.at[idx, 'cka_11_train']
            df.at[idx, 'cka_last_combined_test'] = df.at[idx, 'cka_11_test']
            df.at[idx, 'cka_last_combined_train_ft'] = df.at[idx, 'cka_11_train_ft']
            df.at[idx, 'cka_last_combined_test_ft'] = df.at[idx, 'cka_11_test_ft']
        # ResNet 16 layers (0-16)
        elif 'hiresnet50' in method:
            df.at[idx, 'cka_last_combined_train'] = df.at[idx, 'cka_15_train']
            df.at[idx, 'cka_last_combined_test'] = df.at[idx, 'cka_15_test']
            df.at[idx, 'cka_last_combined_train_ft'] = df.at[idx, 'cka_15_train_ft']
            df.at[idx, 'cka_last_combined_test_ft'] = df.at[idx, 'cka_15_test_ft']

    # Combine ft and fz values into column with _combined suffix
    for col in df.columns:
        if col.endswith('_ft'):
            base_col = col.replace('_ft', '')
            new_col = col.replace('_ft', '_combined')
            df[new_col] = df.apply(
                lambda row: row[base_col] if row['serial'] == 24 else row[col] if row['serial'] == 23 else np.nan,
                axis=1
            )
    
    df = sort_df(df)

    # Calculate logits
    df['acc_2_logit'] = 0.0
    df['acc_2_max_logit'] = 0.0
    df['acc_2_std_logit'] = 0.0

    epsilon = 1e-10
    for idx in df.index:
        # Get acc and acc_max for the current row
        acc = df.at[idx, 'acc_2'] /100
        acc_max = df.at[idx, 'acc_2_max'] /100
        acc_std = df.at[idx, 'acc_2_std'] /100

        clipped_acc = max(min(acc, 1 - epsilon), epsilon)
        clipped_acc_max = max(min(acc_max, 1 - epsilon), epsilon)
        clipped_acc_std = max(min(acc_std, 1 - epsilon), epsilon)

        # logit(p) = log(p / (1 - p))
        df.at[idx, 'acc_2_logit'] = np.log(clipped_acc / (1 - clipped_acc))
        df.at[idx, 'acc_2_max_logit'] = np.log(clipped_acc_max / (1 - clipped_acc_max))
        df.at[idx, 'acc_2_std_logit'] = np.log(clipped_acc_std / (1 - clipped_acc_std))

    df = round_combine_str_mean_std(df, col='acc_2')

    if fp:
        df.to_csv(fp, header=True, index=False)
    return df


def pivot_table(df, serial=None, fp=None, var='acc', rename=True):
    df = df[df['serial'] == serial].copy(deep=False)
    # print(df)
    # df = df.drop_duplicates(subset=['method', 'dataset_name'])

    datasets = [ds for ds in DATASETS_DIC.keys() if ds in df.columns]
    df = df[datasets]


    df['method_order'] = pd.Categorical(df.index, categories=METHODS_DIC.keys(), ordered=True)
    df = df.sort_values(by=['method_order'], ascending=True)
    df = df.drop(columns=['method_order'])


    if rename:
        df.index = df.index.map(rename_var)
        df.columns = df.columns.map(rename_var)


    if fp:
        df.to_csv(fp, header=True, index=True)
    else:
        print(df)


def summarize_results(args):
    # load dataset and preprocess to include method and setting columns, rename val_acc_1 to acc & val_acc_2 to acc_2
    # drop columns
    # filter
    df = preprocess_df(
        args.input_file,
        'acc',
        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
        getattr(args, 'filter_serials', None),
    )

    # aggregate and save results
    fp = os.path.join(args.results_dir, args.output_file)

    # df main uses 'acc', df_2 uses 'acc_2'
    df_main = aggregate_results_main(df, f'{fp}_main.csv', args.main_serials)
    df_2 = aggregate_results_main_2(df, f'{fp}_main_lvl2.csv', args.main_serials)
    print(df_main)
    print(df_2)

    for serial in args.main_serials:
        df_pivoted = pivot_table(df_main, serial, f'{fp}_{serial}_pivoted.csv',
                                 var='acc')
        # pivot_table(df_main, serial, f'{fp}_{serial}_pivoted_mean_std.csv',
        #                          var='acc_mean_std')
        # pivot_table(df_main, serial, f'{fp}_{serial}_pivoted_mean_std_latex.csv',
        #                          var='acc_mean_std_latex')

    return df_main


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str, 
                        default=os.path.join('data', 'hierarchical_all.csv'),
                        help='filename for input .csv file from wandb')

    parser.add_argument('--keep_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--main_serials', nargs='+', type=int, default=[23, 24])

    # output
    parser.add_argument('--output_file', type=str, default='summarized_acc_hierarchical',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'acc_metrics'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    df = summarize_results(args)
    return df


if __name__ == '__main__':
    main()