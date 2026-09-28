import os
import math
import argparse
import numpy as np
import pandas as pd

from utils import preprocess_df, add_all_cols_group, \
    sort_df, sort_cols, DATASETS_DIC, METHODS_DIC


def convert_to_logits(df, col='acc_mean'):
    epsilon = 1e-10

    df[f'{col}_logit'] = df[col].apply(lambda x: math.log(
        (max(min(x / 100, 1 - epsilon), epsilon)) / 
        (1 - (max(min(x / 100, 1 - epsilon), epsilon)))
    ))

    return df


def aggregate_results_all(
    df, acc_col='val_acc_level1', fp=None,
    add_method_avg=True, add_dataset_avg=False,
    group_cols=['serial', 'setting', 'dataset_name', 'method', 'acc_in1k']
):

    # compute method avg (across datasets) or dataset avg (across methods)
    if add_method_avg:
        df = add_all_cols_group(df, 'dataset_name')
    if add_dataset_avg:
        df = add_all_cols_group(df, 'method')


    # columns to aggregate
    kw_list = ['cka_', 'l2_', 'dist_', 'MSC_', 'intra_', 'inter_', 'diversity_']
    agg_cols = [acc_col] + [col for col in df.columns if any(kw in col for kw in kw_list)]
    agg_map = {k: 'mean' for k in agg_cols}

    # compute average, stdev, max, min, ada (deviation to average ratio)
    df_std = df.groupby(group_cols, as_index=False).agg({acc_col: 'std'})
    df_max = df.groupby(group_cols, as_index=False).agg({acc_col: 'max'})
    df_min = df.groupby(group_cols, as_index=False).agg({acc_col: 'min'})
    df = df.groupby(group_cols, as_index=False).agg(agg_map)
    df = df.rename(columns={acc_col: 'acc_mean'})

    df['acc_std'] = df_std[acc_col]
    df['acc_max'] = df_max[acc_col]
    df['acc_min'] = df_min[acc_col]
    df['ada_ratio'] = 100 * (df['acc_std'] / df['acc_mean'])


    # Match ft and fz values into a single column with _matched suffix
    for col in df.columns:
        if col.endswith('_ft'):
            base_col = col.replace('_ft', '')
            new_col = col.replace('_ft', '_matched')
            df[new_col] = df.apply(
                lambda row: row[base_col] if row['serial'] == 24 else row[col] if row['serial'] == 23 else np.nan,
                axis=1
            )


    # convert to logits
    acc_metrics = ['acc_mean', 'acc_std', 'acc_max', 'acc_min']
    for metric in acc_metrics:
        df = convert_to_logits(df, metric)


    # sort dataframe (rows)
    df = sort_df(df)

    # sort dataframe (columns)
    df = sort_cols(df, kw_list=group_cols + ['acc'])


    # save results
    if fp:
        df.to_csv(fp, header=True, index=False)
        print('Saved results to : ', fp)
    return df


def summarize_results(args):
    # load dataset
    df = pd.read_csv(args.input_file)

    # preprocess to include method and setting columns,
    # drop columns
    # filter
    df = preprocess_df(
        df,
        'all',
        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
        getattr(args, 'filter_serials', None),
        getattr(args, 'keep_ratios', None),
        getattr(args, 'keep_extractor', None),
        getattr(args, 'keep_ipc', None),
    )

    # aggregate and save results
    fp = os.path.join(args.results_dir, args.output_file)

    acc_cols = ['val_acc_level1', 'val_acc_level2', 'ap_w']

    for acc_col in acc_cols:
        fn = f'{fp}_{acc_col}'
        df_main = aggregate_results_all(df, acc_col, f'{fn}_all.csv')
        print(df_main)

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
    parser.add_argument('--output_file', type=str, default='summary',
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