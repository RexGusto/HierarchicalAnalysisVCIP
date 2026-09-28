import os
import argparse

import pandas as pd

from utils import preprocess_df, sort_df, group_by_family


REASSIGN_DIC = {
    41: 23,
    43: 24,
}


def reassign_serial(df):
    df['serial'] = df['serial'].apply(lambda x: REASSIGN_DIC.get(x, x))
    return df


def agg_train_flops(df, ds='cub'):
    if ds:
        df = df[(df['dataset_name'] == ds)].copy(deep=False)

    df = df.groupby(['serial', 'method'], as_index=False).agg({
        'flops': 'mean',
    })
    return df


def summarize_train_flops(input_file, keep_serials=None):
    # read and preprocess
    df = pd.read_csv(input_file)

    df = preprocess_df(
        df,
        'train_cost',
        keep_serials=keep_serials,
    )


    # aggregate parameters and flops and combine
    df = agg_train_flops(df)


    # keep relevant columns
    cols_keep = ['serial', 'method', 'flops',]
    df = df[cols_keep]


    # to be consistent with the naming criteria in the other files / results
    df = reassign_serial(df)

    return df


def agg_parameters(df):
    # aggregate across seeds
    df = df.groupby(['serial', 'method', 'dataset_name'], as_index=False).agg({
       'no_params': 'mean',
       'no_params_trainable': 'mean',
    })


    # mean aggregate across datasets
    df_agg_mean = df.groupby(['serial', 'method'], as_index=False).agg({
        'no_params': 'mean',
        'no_params_trainable': 'mean',
    })


    # sum aggregate across datasets
    df_agg_sum = df.groupby(['serial', 'method'], as_index=False).agg({
        'no_params': 'sum',
        'no_params_trainable': 'sum',
    })

    renames = {
        'no_params': 'no_params_total',
        'no_params_trainable': 'no_params_trainable_total'
    }
    df_agg_sum = df_agg_sum.rename(columns=renames)


    # combine mean and total aggregates
    df = pd.merge(df_agg_mean, df_agg_sum,
                  how='left', on=['serial', 'method'])


    # percentage of trainable parameters
    df['trainable_percent'] = 100 * (df['no_params_trainable'] / df['no_params'])

    return df


def summarize_parameters(input_file, keep_datasets=None, keep_methods=None,
    keep_serials=None):
    df = pd.read_csv(input_file)

    df = preprocess_df(
        df,
        'train_cost',
        keep_datasets,
        keep_methods,
        keep_serials,
    )

    # aggregate parameters
    df = agg_parameters(df)

    # keep relevant columns
    cols_keep = ['serial', 'method',
                 'trainable_percent', 'no_params', 'no_params_trainable',
                 'no_params_total', 'no_params_trainable_total']
    df = df[cols_keep]

    return df


def summarize_acc(input_file, acc_col='val_acc_level1', keep_datasets=None,
                  keep_methods=None, keep_serials=None,
                  group_keys=['serial', 'setting', 'method']):
    df = pd.read_csv(input_file)

    df = preprocess_df(
        df,
        'acc',
        keep_datasets,
        keep_methods,
        keep_serials,
    )

    df_std = df.groupby(group_keys, as_index=False).agg({acc_col: 'std'})
    df_max = df.groupby(group_keys, as_index=False).agg({acc_col: 'max'})
    df = df.groupby(group_keys, as_index=False).agg({acc_col: 'mean'})
    df = df.rename(columns={acc_col: 'acc_mean'})

    df['acc_std'] = df_std[acc_col]
    df['acc_max'] = df_max[acc_col]
    df['ada_ratio'] = 100 * (df['acc_std'] / df['acc_mean'])

    return df



def summarize_acc_cost(args):
    # read acc
    df_acc = summarize_acc(
        args.input_file_acc, args.acc_to_use,
        getattr(args, 'keep_datasets', None), getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None))


    # read train cost
    df_parameters = summarize_parameters(args.input_file_acc,
        getattr(args, 'keep_datasets', None), getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None))

    df_train_flops = summarize_train_flops(args.input_file_train_cost,
        getattr(args, 'keep_serials', None))


    # combine acc, train and test cost and sort based on method
    # outer if want to keep even if not all match
    df = pd.merge(df_acc, df_parameters, how='left', on=['serial', 'method'])
    df = pd.merge(df, df_train_flops, how='left', on=['serial', 'method'])
    df = sort_df(df, method_only=True)


    # add column that groups up into ResNet/ViTs
    df['family'] = df['method'].apply(group_by_family)


    # aggregate and save results
    fp = os.path.join(args.results_dir, f'{args.output_file}.csv')
    df.to_csv(fp, header=True, index=False)

    return df


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file_acc', type=str, 
                        default=os.path.join('data', 'hierarchical_all.csv'),
                        help='filename for input .csv file from wandb')
    parser.add_argument('--input_file_train_cost', type=str,
                        default=os.path.join('data', 'hierarchical_all.csv'),
                        help='filename for input .csv file from wandb')

    parser.add_argument('--keep_datasets', nargs='+', type=str,
                        default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str,
                        default=None)
    parser.add_argument('--keep_serials', nargs='+', type=int, default=[23, 24])

    parser.add_argument('--host', type=str, default='server-3090')

    parser.add_argument('--acc_to_use', type=str, default='val_acc_level1')

    # output
    parser.add_argument('--output_file', type=str, default='cost_val_acc_level1',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'cost'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    df = summarize_acc_cost(args)
    return df


if __name__ == '__main__':
    main()
