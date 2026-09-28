import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import filter_df, preprocess_df
from utils import add_setting
from utils import METHODS_DIC
from codebase.plots.plot import make_plot


def stats_analysis(df):
    df = df[~df.isnull().values.any(axis=1)].copy(deep=False)

    x = df['lr']
    x = sm.add_constant(x)
    y = df['val_acc_level1']

    # results = smf.ols('val_acc ~ lr', data=df).fit()
    results = sm.OLS(y, x).fit()

    print(results.summary())
    return 0


def compute_ada_ratio(df, args):
    df = df.copy(deep=False)
    df_oth = df.copy(deep=False)

    df_std = df.groupby(['serial','dataset_name', 'setting', 'method'], as_index=False)['val_acc_level1'].std(numeric_only=True)
    df = df.groupby(['serial','dataset_name', 'setting', 'method'], as_index=False)['val_acc_level1'].mean(numeric_only=True)
    # df_std = df.groupby(['serial', 'setting', 'method'], as_index=False)['val_acc_level1'].std(numeric_only=True)
    # df = df.groupby(['serial', 'setting', 'method'], as_index=False)['val_acc_level1'].mean(numeric_only=True)
    df['std'] = df_std['val_acc_level1']
    df.rename(columns={'val_acc_level1': 'mean'}, inplace=True)

    df['ada_ratio'] = 100 * (df['std'] / df['mean'])
    df = df[['serial','dataset_name', 'setting', 'method', 'ada_ratio', 'mean', 'std']]
    # df = df[['serial', 'setting', 'method', 'ada_ratio', 'mean', 'std']]

    # add per dataset averages / stds as rows (method = 'all_mean/std')
    # averages of all models (frozen , finetuned and cal)
    dataset_avgs = df_oth.groupby(['serial','dataset_name', 'setting'], as_index=False).mean(numeric_only=True)
    dataset_avgs['method'] = 'all_mean'
    dataset_stds = df_oth.groupby(['serial','dataset_name', 'setting'], as_index=False).std(numeric_only=True)
    dataset_stds['method'] = 'all_std'

    # add per model averages as rows (dataset_name = 'all_mean/std')
    model_avgs = df.groupby(['serial','method','setting'], as_index=False).mean(numeric_only=True)
    model_avgs['dataset_name'] = 'all_mean'
    # Step 2: Create base method column
    model_avgs['method_base'] = model_avgs['method'].str.replace('_fz$', '', regex=True)

    # Step 3: Compute mean ada_ratio for each base method
    base_model_means = model_avgs.groupby('method_base', as_index=False)['ada_ratio'].mean(numeric_only=True)

    # Step 4: Create rows to append
    base_model_means['serial'] = 255
    base_model_means['dataset_name'] = 'all_mean'
    base_model_means['method'] = base_model_means['method_base']
    base_model_means['setting'] = np.nan
    base_model_means['mean'] = np.nan
    base_model_means['std'] = np.nan

    # Only keep the required columns in the correct order
    summary_rows = base_model_means[['serial', 'dataset_name', 'method', 'setting', 'ada_ratio', 'mean', 'std']]

    # Step 5: Drop helper column and append new rows
    model_avgs.drop(columns=['method_base'], inplace=True)
    model_avgs = pd.concat([model_avgs, summary_rows], axis=0, ignore_index=True)
        
    model_stds = df.groupby(['serial','method','setting'], as_index=False).std(numeric_only=True)
    model_stds['dataset_name'] = 'all_std'

    # Work only with all_mean rows (from model_avgs)
    df_all_mean = model_avgs.copy(deep=True)
    df_all_mean['method_base'] = df_all_mean['method'].str.replace('_fz$', '', regex=True)

    # Pivot into wide format
    ada_matrix = df_all_mean.pivot_table(
        index='serial',
        columns='method_base',
        values='ada_ratio',
        aggfunc='first'
    )

    # Use serial 255 to determine column order
    method_order = (
        df_all_mean[df_all_mean['serial'] == 255]
        .sort_values(by='ada_ratio')  # optional, consistent sorting
        ['method_base']
        .tolist()
    )
    ada_matrix = ada_matrix.reindex(columns=method_order)

    # Apply readable names to columns using METHODS_DIC (on base method names)
    display_name_map = {}
    for k, v in METHODS_DIC.items():
        base = k.replace('_fz', '')
        display_name_map[base] = v  # last overwrite wins, fine for same name pairs

    # Rename columns (method_base → readable name)
    ada_matrix.columns = [display_name_map.get(col, col) for col in ada_matrix.columns]

    ada_matrix = ada_matrix.round(2)
    
    # Save to CSV
    matrix_file = os.path.join(args.results_dir, f'{args.output_file}_ada_matrix.csv')
    ada_matrix.to_csv(matrix_file)

    df_concat = pd.concat([df, dataset_avgs, dataset_stds, model_avgs, model_stds], axis=0)

    output_file = os.path.join(args.results_dir, f'{args.output_file}.csv')
    # df.to_csv(output_file, header=True, index=False)

    # df_concat is to create the csv with the 'all_mean' & 'all_std'
    # df is used for the subset to avoid index overlapping issue for the subset making
    df_concat.to_csv(output_file, header=True, index=False)

    model_avgs.to_csv(output_file.replace('.csv', '_models.csv'), header=True, index=False)
    dataset_avgs.to_csv(output_file.replace('.csv', '_datasets.csv'), header=True, index=False)

    return df


def make_acc_dist_subset(args, df, df_ada):
    # alternatively could use some linearly space (np.linspace(0, len(df_ada), n=5)
    cols = ['dataset_name', 'method', 'serial']
    cols_acc = cols + ['val_acc_level1']
    # print(cols_acc)

    df_ada_sorted = df_ada.sort_values(by=['ada_ratio'], ascending=True).reset_index(drop=True)

    if args.linspace_k:
        indexes = np.linspace(0, len(df_ada_sorted) - 1, args.linspace_k, dtype=int)
        low_index = (len(df_ada_sorted) - 1) // 3
        high_index = (len(df_ada_sorted) - 1) - (len(df_ada_sorted) - 1) // 3

        subsets = pd.DataFrame()

        for i in indexes:
            ds, model, setting = df_ada_sorted.loc[i, cols]

            subset = df[(df['dataset_name'] == ds) &
                            (df['method'] == model) &
                            (df['serial'] == setting)][cols_acc]
            subset['ada_ratio'] = df_ada_sorted.loc[i, 'ada_ratio']

            if i <= low_index:
                subset['Ratio'] = 'Low'
            elif i >= high_index:
                subset['Ratio'] = 'High'
            else:
                subset['Ratio'] = 'Medium'

            subsets = pd.concat([subsets, subset], axis=0)

    else:
        median_idx = len(df_ada_sorted) // 2
        max_idx = df_ada['ada_ratio'].idxmax()
        min_idx = df_ada['ada_ratio'].idxmin()

        median_ds, median_model, median_setting = df_ada_sorted.loc[median_idx, cols]
        max_ds, max_model, max_setting = df_ada.loc[max_idx, cols]
        min_ds, min_model, min_setting = df_ada.loc[min_idx, cols]
        # print(df_ada.loc[20, cols])

        subset_median = df[(df['dataset_name'] == median_ds) &
                        (df['method'] == median_model) &
                        (df['serial'] == median_setting)][cols_acc]
        subset_median['Ratio'] = 'Median'
        subset_median['ada_ratio'] = df_ada_sorted.loc[median_idx, 'ada_ratio']

        subset_max = df[(df['dataset_name'] == max_ds) &
                        (df['method'] == max_model) &
                        (df['serial'] == max_setting)][cols_acc]
        subset_max['Ratio'] = 'Max'
        subset_max['ada_ratio'] = df_ada.loc[max_idx, 'ada_ratio']
        # print(min_ds)
        subset_min = df[(df['dataset_name'] == min_ds) &
                        (df['method'] == min_model) &
                        (df['serial'] == min_setting)][cols_acc]
        subset_min['Ratio'] = 'Min'
        subset_min['ada_ratio'] = df_ada.loc[min_idx, 'ada_ratio']
        # print(subset_min)

        subsets = pd.concat([subset_min, subset_median, subset_max], axis=0)

    subsets['ada_ratio'] = subsets['ada_ratio'].apply(lambda x: round(x, 2))

    print(subsets)
    # print(subset_max)
    # print(subset_min)
    # print(subset_median)
    # print(df_ada.loc[max_idx])
    # print(df_ada.loc[df_ada['ada_ratio'].idxmin()])
    # print(df_ada_sorted.loc[len(df_ada_sorted) // 2])

    return subsets

# def make_acc_dist_subset(args, df, df_ada):
#     cols = ['dataset_name', 'method', 'serial']
#     cols_acc = cols + ['val_acc_level1']

#     df_ada_sorted = df_ada.sort_values(by=['ada_ratio'], ascending=True).reset_index(drop=True)

#     if args.linspace_k:
#         indexes = np.linspace(0, len(df_ada_sorted) - 1, args.linspace_k, dtype=int)
#         low_index = (len(df_ada_sorted) - 1) // 3
#         high_index = (len(df_ada_sorted) - 1) - (len(df_ada_sorted) - 1) // 3

#         subsets = pd.DataFrame()

#         for i in indexes:
#             # Since df_ada is now per-model, it doesn't have 'dataset_name'
#             model, setting = df_ada_sorted.loc[i, ['method', 'serial']]
#             # Use all datasets for the given model and setting
#             subset = df[(df['method'] == model) &
#                         (df['serial'] == setting)][cols_acc]
#             subset['ada_ratio'] = df_ada_sorted.loc[i, 'ada_ratio']

#             if i <= low_index:
#                 subset['Ratio'] = 'Low'
#             elif i >= high_index:
#                 subset['Ratio'] = 'High'
#             else:
#                 subset['Ratio'] = 'Medium'

#             subsets = pd.concat([subsets, subset], axis=0)

#     else:
#         median_idx = len(df_ada_sorted) // 2
#         max_idx = df_ada['ada_ratio'].idxmax()
#         min_idx = df_ada['ada_ratio'].idxmin()

#         median_model, median_setting = df_ada_sorted.loc[median_idx, ['method', 'serial']]
#         max_model, max_setting = df_ada.loc[max_idx, ['method', 'serial']]
#         min_model, min_setting = df_ada.loc[min_idx, ['method', 'serial']]

#         subset_median = df[(df['method'] == median_model) &
#                            (df['serial'] == median_setting)][cols_acc]
#         subset_median['Ratio'] = 'Median'
#         subset_median['ada_ratio'] = df_ada_sorted.loc[median_idx, 'ada_ratio']

#         subset_max = df[(df['method'] == max_model) &
#                         (df['serial'] == max_setting)][cols_acc]
#         subset_max['Ratio'] = 'Max'
#         subset_max['ada_ratio'] = df_ada.loc[max_idx, 'ada_ratio']

#         subset_min = df[(df['method'] == min_model) &
#                         (df['serial'] == min_setting)][cols_acc]
#         subset_min['Ratio'] = 'Min'
#         subset_min['ada_ratio'] = df_ada.loc[min_idx, 'ada_ratio']

#         subsets = pd.concat([subset_min, subset_median, subset_max], axis=0)

#     subsets['ada_ratio'] = subsets['ada_ratio'].apply(lambda x: round(x, 2))

#     return subsets


def analyze_ada(args):
    df = pd.read_csv(args.input_file_stage1)
    # df = add_setting(df)

    df = preprocess_df(
        df, 'acc',
        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
        getattr(args, 'filter_serials', None),
    )

    df = df[['serial', 'dataset_name', 'setting', 'method', 'val_acc_level1', 'lr']]

    # stats_analysis(df)

    print(df)
    stats_analysis(df)
    df_ada = compute_ada_ratio(df, args)
    print(df_ada)

    subsets = make_acc_dist_subset(args, df, df_ada)
    make_plot(args, subsets)

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file_stage1', type=str,
                        default=os.path.join('data', 'hierarchical_all.csv'),
                        help='filename for input.csv file from wandb')

    parser.add_argument('--filter_og_only', action='store_true', help='')

    parser.add_argument('--linspace_k', type=int, default=0,
                        help='plot k linearly spaced acc dist based on ada ratio')
    # output
    parser.add_argument('--results_dir',type=str,
                        default=os.path.join('results_all', 'ada_ratio'),
                        help='The directory where results will be stored')
    parser.add_argument('--output_file', default='ada_ratio', type=str, help='File path')
    parser.add_argument('--save_format', choices=['pdf', 'png', 'jpg'], default='png', type=str,
                        help='Print stats on word level if use this command')

    parser.add_argument('--type_plot', default='box',
                        help='the type of plot (line, bar)')
    parser.add_argument('--x_var_name', type=str, default='ada_ratio',
                        help='name of the variable for x')
    parser.add_argument('--y_var_name', type=str, default='val_acc_level1',
                        help='name of the variable for y')
    parser.add_argument('--hue_var_name', type=str, default=None,
                        help='legend of this bar plot')
    # style related
    parser.add_argument('--context', type=str, default='notebook',
                        help='''affects font sizes and line widths
                        # notebook (def), paper (small), talk (med), poster (large)''')
    parser.add_argument('--style', type=str, default='whitegrid',
                        help='''affects plot bg color, grid and ticks
                        # whitegrid (white bg with grids), 'white', 'darkgrid', 'ticks'
                        ''')
    parser.add_argument('--palette', type=str, default='colorblind',
                        help='''
                        color palette (overwritten by color)
                        # None (def), 'pastel', 'Blues' (blue tones), 'colorblind'
                        # can create a palette that highlights based on a category
                        can create palette based on conditions
                        pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}
                        pal = {species: "r" if species == "versicolor" else "b" for species in df.species.unique()}
                        ''')
    parser.add_argument('--color', type=str, default='tab:blue')
    parser.add_argument('--font_family', type=str, default='sans-serif',
                        help='font family (sans-serif or serif)')
    parser.add_argument('--font_scale', type=int, default=1.0,
                        help='adjust the scale of the fonts')
    parser.add_argument('--bg_line_width', type=int, default=0.25,
                    help='adjust the scale of the line widths')
    parser.add_argument('--line_width', type=int, default=1.0,
                        help='adjust the scale of the line widths')
    parser.add_argument('--fig_size', nargs='+', type=float, default=[6, 6],
                        help='size of the plot')
    parser.add_argument('--dpi', type=int, default=300)

    # Set title, labels and ticks
    parser.add_argument('--log_scale_x', action='store_true')
    parser.add_argument('--log_scale_y', action='store_true')
    parser.add_argument('--title', type=str,
                        default='Accuracy Distribution for Different ADA Ratios',
                        help='title of the plot')
    parser.add_argument('--x_label', type=str, default='Accuracy Deviation-Average Ratio (%)',
                        help='x label of the plot')
    parser.add_argument('--y_label', type=str, default='Top-1 Accuracy (%)',
                        help='y label of the plot')
    parser.add_argument('--y_lim', nargs='*', type=int, default=[0, 100],
                        help='limits for y axis (suggest --ylim 0 100)')
    parser.add_argument('--xticks_labels', nargs='+', type=str, default=None,
                        help='labels of x-axis ticks')
    parser.add_argument('--x_ticks', nargs='+', type=int, default=None)
    parser.add_argument('--x_ticks_labels', nargs='+', type=str, default=None,
                        help='labels of x-axis ticks')
    parser.add_argument('--x_rotation', type=int, default=None,
                        help='lotation of x-axis lables')
    parser.add_argument('--y_rotation', type=int, default=None,
                        help='lotation of y-axis lables')
    
    parser.add_argument('--keep_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--filter_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--filter_methods', nargs='+', type=str, default=None)

    # Change location of legend
    parser.add_argument('--loc_legend', type=str, default='lower left',
                        help='location of legend options are upper, lower, left right, center')


    args= parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.color:
        args.palette = [args.color for _ in range(100)]

    analyze_ada(args)

    return 0


if __name__ == '__main__':
    main()