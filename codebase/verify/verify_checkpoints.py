import os
import sys
import glob
import argparse

import pandas as pd
import torch


# num_images_train separates train from train+val runs and few-example runs
KEYS = ['serial', 'dataset_name', 'model_name', 'model_name_extractor', 'n_cluster_ratio', 'opt', 'lr', 'seed', 'num_images_train']


def read_ckpt_meta(path):
    ckpt = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    cfg = vars(ckpt['config'])
    meta = {k: cfg.get(k) for k in KEYS + ['train_trainval', 'epochs', 'image_size', 'freeze_backbone']}
    meta.update(path=path, epoch=ckpt.get('epoch'), ckpt_acc=ckpt.get('accuracy'))
    return meta


def load_runs(data_dir):
    # the run exports only; wandb_runs.csv is the project index, matched in wandb_match.csv
    files = [f for f in sorted(glob.glob(os.path.join(data_dir, '*.csv'))) if os.path.basename(f) != 'wandb_runs.csv']
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat([d for d in dfs if 'serial' in d.columns], ignore_index=True)
    df = df[KEYS + ['val_acc_level1']].drop_duplicates()
    return df


def match(ckpts, runs):
    for df in (ckpts, runs):
        df['n_cluster_ratio'] = pd.to_numeric(df['n_cluster_ratio'], errors='coerce').fillna(0).astype(int)
        # runs without a pseudo-hierarchy have no extractor; a pseudo run's extractor defaults to its own model
        df['model_name_extractor'] = df['model_name_extractor'].fillna(df['model_name']).where(df['n_cluster_ratio'] > 0, '')
        df['lr'] = pd.to_numeric(df['lr']).round(10)
    runs = runs.groupby(KEYS, as_index=False).agg(csv_acc=('val_acc_level1', 'max'), csv_rows=('val_acc_level1', 'size'))
    out = ckpts.merge(runs, on=KEYS, how='left')
    assert len(out) == len(ckpts)
    out['acc_diff'] = (out['ckpt_acc'] - out['csv_acc']).round(4)
    return out


def load_test(path, code_repo):
    sys.path.insert(0, code_repo)
    from hierarchical.model_utils.build_model import TIMM
    from hierarchical.other_utils.build_args import parse_train_args
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    # configs saved before an option existed take the code repo's default
    for k, v in vars(parse_train_args(ret_parser=True).parse_args([])).items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.pretrained = False
    model = TIMM(cfg)
    model.load_state_dict(ckpt['model'], strict=True)
    return True


def main():
    parser = argparse.ArgumentParser(description='Match saved checkpoints to their wandb rows; strict-load a sample')
    parser.add_argument('--ckpt_root', type=str, required=True)
    parser.add_argument('--code_repo', type=str, required=True, help='projects/Hierarchical checkout')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--load_per_folder', type=int, default=1)
    parser.add_argument('--results_dir', type=str, default=os.path.join('results_all', 'checkpoints'))
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.ckpt_root, '**', '*.pth'), recursive=True))
    print(f'{len(paths)} checkpoints under {args.ckpt_root}')
    ckpts = pd.DataFrame([read_ckpt_meta(p) for p in paths])
    ckpts['folder'] = ckpts['path'].map(lambda p: os.path.relpath(os.path.dirname(p), args.ckpt_root))

    out = match(ckpts, load_runs(args.data_dir))
    out['load_strict'] = ''
    for folder, g in out.groupby('folder'):
        for i in g.index[:args.load_per_folder]:
            try:
                out.loc[i, 'load_strict'] = str(load_test(out.loc[i, 'path'], args.code_repo))
            except Exception as e:
                out.loc[i, 'load_strict'] = f'{type(e).__name__}: {str(e)[:200]}'
            print(folder, os.path.basename(out.loc[i, 'path']), out.loc[i, 'load_strict'])

    out['path'] = out['path'].map(lambda p: os.path.relpath(p, args.ckpt_root))
    os.makedirs(args.results_dir, exist_ok=True)
    out.to_csv(os.path.join(args.results_dir, 'checkpoint_inventory.csv'), index=False)

    summary = out.groupby('folder').agg(
        ckpts=('path', 'size'), serials=('serial', lambda s: ','.join(map(str, sorted(s.unique())))),
        matched=('csv_acc', lambda s: int(s.notna().sum())),
        acc_equal=('acc_diff', lambda s: int((s.abs() < 0.01).sum())),
        max_abs_diff=('acc_diff', lambda s: s.abs().max()),
        loaded=('load_strict', lambda s: f"{(s == 'True').sum()}/{(s != '').sum()}")).reset_index()
    summary.to_csv(os.path.join(args.results_dir, 'checkpoint_summary.csv'), index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
