import os
import re
import argparse
import pandas as pd

from codebase.download.download_save_wandb_data import get_wandb_project_runs


CONFIG_COLS = ['dataset_name', 'model_name', 'opt', 'lr', 'seed', 'n_cluster_ratio', 'use_hierarchy', 'cfg',
               'df_train', 'df_trainval', 'train_trainval', 'epochs', 'image_size']


def repo_name(remote):
    return re.sub(r'^.*github\.com[:/]|\.git$', '', remote) if remote else None


def repo_path(program):
    # tools/train.py, not the host's absolute path
    return '/'.join(program.split('/')[-2:]) if program else None


def make_df(runs):
    rows = []
    for run in runs:
        meta = run.metadata or {}
        git = meta.get('git') or {}
        row = {'run_id': run.id, 'run_name': run.name, 'state': run.state, 'created_at': run.created_at,
               'serial': run.config.get('serial'), 'repo': repo_name(git.get('remote')), 'commit': git.get('commit'),
               'host': meta.get('host'), 'program': repo_path(meta.get('program')), 'args': ' '.join(meta.get('args') or [])}
        row.update({c: run.config.get(c) for c in CONFIG_COLS})
        summary = run.summary._json_dict
        row.update(val_acc_level1=summary.get('val_acc_level1'), num_images_train=summary.get('num_images_train'))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['dataset_name', 'model_name', 'opt', 'lr', 'created_at'])


def main():
    parser = argparse.ArgumentParser(description="Save each run's launch args, code commit and key config for given serials")
    parser.add_argument('--serials', nargs='+', type=int, required=True)
    parser.add_argument('--project_name', type=str, default='nycu_pcs/Hierarchical')
    parser.add_argument('--results_dir', type=str, default=os.path.join('data', 'run_args'))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    for serial in args.serials:
        df = make_df(get_wandb_project_runs(args.project_name, [serial]))
        fp = os.path.join(args.results_dir, f'serial_{serial}.csv')
        df.to_csv(fp, index=False)
        print(f'serial {serial}: {len(df)} runs -> {fp}')


if __name__ == '__main__':
    main()
