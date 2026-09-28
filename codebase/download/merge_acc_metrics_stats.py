import os
import argparse
import numpy as np
import pandas as pd


SORT_COLS = [
    'serial', 'dataset_name', 'model_name',
    'lr', 'seed', 'host', 'batch_size',
]


def load_process_pretraining_stats(input_file):
    # Load the two CSV files
    df_stats = pd.read_csv(input_file)

    # Add "hi-" prefix to the 'pt' column in both DataFrames
    df_stats['model_name'] = 'hi' + df_stats['pt'].astype(str)

    # Keep only the relevant columns: 'pt', 'augreg', 'acc_in1k'
    df_stats = df_stats[['model_name', 'augreg', 'acc_in1k']]
    return df_stats


def merge_pretraining_stats(df, stats_df):
    # Merge the W&B DataFrame with the pretraining stats DataFrame
    # Match 'model_name' from df with 'pt' from stats_df
    df = df.merge(stats_df, how='left', left_on='model_name', right_on='pt')
    
    # Drop the redundant 'pt' column after merging
    if 'pt' in df.columns:
        df = df.drop(columns=['pt'])

    return df


def compute_cis(df):
    # Compute CIS = diversity * acc_in1k
    # sources: frozen model metrics ('') or finetuned model metrics ('ft')
    sources = ['', '_ft']
    splits = ['train', 'test']

    # for source in sources:
    #     for split in splits:
    #         df[f'cka_inv_low_mean_{split}{source}'] = (df[f'cka_low_mean_{split}{source}'] * -1) + 1
    #         df[f'cka_inv_high_mean_{split}{source}'] = (df[f'cka_high_mean_{split}{source}'] * -1) + 1

    # metrics = ['spectral_diversity', 'clustering_diversity', 'cka_0', 'cka_last',
    #         'dist_0', 'dist_last', 'cka_low_mean', 'cka_high_mean', 'cka_inv_low_mean', 'cka_inv_high_mean']
    
    metrics = ['spectral_diversity', 'clustering_diversity']
    
    for source in sources:
        for m in metrics:
            for split in splits:
                df[f'cis_{m}_{split}{source}'] = df[f'{m}_{split}{source}'] * df['acc_in1k']
    return df


# def assign_metric_last_layer(df):
#     # sources: frozen model metrics ('') or finetuned model metrics ('ft')
#     sources = ['', '_ft']
#     metrics = ['cka', 'dist']
#     splits = ['train', 'test']

#     for source in sources:
#         for m in metrics:
#             for split in splits:
#                 col_name = f'{m}_last_{split}{source}'
#                 df[col_name] = np.nan

#                 # Masks based on the 'model_name' column
#                 mask_vit = df['model_name'].str.contains('hivit|hideit', case=False, na=False)
#                 mask_resnet = df['model_name'].str.contains('hiresnet50', case=False, na=False)

#                 # Assign values for ViT models (layer 11)
#                 df.loc[mask_vit, col_name] = df.loc[mask_vit, f'{m}_11_{split}{source}']

#                 # Assign values for ResNet models (layer 15)
#                 df.loc[mask_resnet, col_name] = df.loc[mask_resnet, f'{m}_15_{split}{source}']

#     return df


def sort_save_df(df, fp, sort_cols=['serial']):
    df = df.sort_values(by=sort_cols, ascending=[True for _ in sort_cols])
    df.to_csv(fp, header=True, index=False)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--input_file_acc', type=str,
                        default=os.path.join('data', 'hierarchical_stage1.csv'),
                        help='filename for input .csv file')
    parser.add_argument('--input_file_metrics', type=str,
                        default=os.path.join('data', 'hierarchical_feature_metrics.csv'),
                        help='filename for input .csv file')
    parser.add_argument('--input_file_pretrainings', type=str,
                        default=os.path.join('data', 'stats_pretrainings.csv'))

    # Output
    parser.add_argument('--output_file', default='hierarchical_all_1.csv', type=str,
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

    # load dataframes
    df = pd.read_csv(args.input_file_acc)
    df_metrics = pd.read_csv(args.input_file_metrics)
    stats_df = load_process_pretraining_stats(args.input_file_pretrainings)
    print(len(df), len(df.columns), list(df.columns), df.iloc[0])

    # merge dataframes
    df = pd.merge(df, df_metrics, how='left', on=['dataset_name', 'model_name'])
    df = pd.merge(df, stats_df, how='left', on=['model_name'])
    print(len(df), len(df.columns), list(df.columns), df.iloc[0])

    # compute new columns
    # df = assign_metric_last_layer(df)
    # df = compute_cis(df)
    print(len(df), len(df.columns), list(df.columns), df.iloc[0])

    # Sort and save the updated DataFrame
    sort_save_df(df, args.output_file, args.sort_cols)

    return 0

if __name__ == '__main__':
    main()