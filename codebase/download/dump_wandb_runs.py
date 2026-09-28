import os
import argparse
import pandas as pd

from codebase.download.download_save_wandb_data import get_wandb_project_runs


CONFIG_COLS = [
    'serial', 'dataset_name', 'model_name', 'opt', 'lr', 'seed', 'epochs', 'image_size', 'batch_size',
    'use_hierarchy', 'n_cluster_ratio', 'model_name_extractor', 'freeze_backbone', 'train_trainval',
    'num_images_train', 'img_per_class', 'head_design',
]
SUMMARY_COLS = ['val_acc_level1', 'ap_w']


def make_df(runs):
    rows = []
    for run in runs:
        row = {'run_id': run.id, 'run_name': run.name, 'state': run.state, 'created_at': run.created_at}
        row.update({c: run.config.get(c) for c in CONFIG_COLS})
        # the paged listing already holds the summary; dict(run.summary) costs one request per run
        summary = run.summary._json_dict
        row.update({c: summary.get(c) for c in SUMMARY_COLS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['serial', 'created_at'])


def parse_args():
    parser = argparse.ArgumentParser(description='Index every run of the wandb project: id, state, key config, final accuracy')
    parser.add_argument('--project_name', type=str, default='nycu_pcs/Hierarchical')
    parser.add_argument('--output_file', type=str, default=os.path.join('data', 'wandb_runs.csv'))
    return parser.parse_args()


def main():
    args = parse_args()
    df = make_df(get_wandb_project_runs(args.project_name))
    df.to_csv(args.output_file, index=False)
    print(f'{len(df)} runs -> {args.output_file}')


if __name__ == '__main__':
    main()
