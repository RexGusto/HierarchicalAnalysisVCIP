import os
import argparse
import numpy as np
import pandas as pd

from utils import preprocess_df, filter_df, add_all_cols_group, \
    round_combine_str_mean_std, sort_df, rename_var, \
    DATASETS_UFGIR, DATASETS_DIC, METHODS_DIC
from codebase.plots.plot import make_plot

def compute_z_scores(grouped_df, columns):
    z_scores = {}
    for col in columns:
        mean = grouped_df[col].mean()
        std = grouped_df[col].std()
        z_scores[col] = (grouped_df[col] - mean) / std

    result = pd.DataFrame(z_scores)
    result['method'] = grouped_df['method']

    return result

def combine_z_score(df):
    result = []
    serials = df['serial'].unique()
    datasets = df['dataset_name'].unique()

    for serial in serials:
        serial_df = df[df['serial'] == serial]
        for dataset in datasets:
            dataset_df = serial_df[serial_df['dataset_name'] == dataset]
            z_score_df = compute_z_scores(dataset_df, ['acc_mean', 'acc_std', 'acc_max', 'acc_min'])

            combined_df = dataset_df[['serial', 'dataset_name', 'method', 'acc_mean', 'acc_std', 'acc_max', 'acc_min']].copy()
            combined_df = combined_df.merge(
                z_score_df[['method', 'acc_mean', 'acc_std', 'acc_max', 'acc_min']],
                on='method',
                suffixes=('', '_z_score')
            )
            combined_df['penalized_acc'] = combined_df['acc_max'] - combined_df['acc_std_z_score']

            result.append(combined_df)

    final_result = pd.concat(result, ignore_index=True)
    return final_result


def summarize_results(args):
    df = pd.read_csv(args.input_file)

    df = filter_df(
        df,
        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
    )

    fp = os.path.join(args.results_dir, args.output_file)

    df_main = combine_z_score(df)
    df_main.to_csv(f'{fp}_main.csv', header=True, index=False)
    

def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str, 
                        default=os.path.join('results_all', 'acc',
                                             'summary_val_acc_level1_main.csv'),
                        help='filename for input .csv file from wandb')

    parser.add_argument('--keep_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--filter_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--filter_methods', nargs='+', type=str, default=None)
    parser.add_argument('--keep_serials', nargs='+', type=int, default=[23, 24])

    # output
    parser.add_argument('--output_file', type=str, default='z_score',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'z_score'),
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
