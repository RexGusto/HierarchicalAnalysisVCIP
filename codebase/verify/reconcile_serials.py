import os
import re
import glob
import argparse

import pandas as pd


def documented_serials(path):
    # first column of every table row in SERIALS.md; "42–49" and "56–61" are ranges
    serials = set()
    for line in open(path):
        if not line.startswith('| ') or line.startswith('| Serial'):
            continue
        cell = line.split('|')[1]
        for a, b in re.findall(r'(\d+)\s*[–-]\s*(\d+)', cell):
            serials.update(range(int(a), int(b) + 1))
        serials.update(int(x) for x in re.findall(r'\d+', cell))
    return serials


def csv_rows(data_dir):
    # rows per serial in the CSV that holds most of them; files overlap, so a union would double count
    counts = {}
    for f in glob.glob(os.path.join(data_dir, '*.csv')):
        d = pd.read_csv(f, low_memory=False)
        if 'serial' not in d or 'run_id' in d:
            continue
        for s, n in d.groupby('serial').size().items():
            if n > counts.get(s, (0, ''))[0]:
                counts[s] = (n, os.path.basename(f))
    return counts


def main():
    parser = argparse.ArgumentParser(description='Per serial: wandb runs against committed CSV rows, SERIALS.md and checkpoints')
    parser.add_argument('--wandb_runs', type=str, default=os.path.join('data', 'wandb_runs.csv'))
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--serials_doc', type=str, default=os.path.join('docs', 'SERIALS.md'))
    parser.add_argument('--checkpoints', type=str, default=os.path.join('results_all', 'checkpoints'))
    parser.add_argument('--output_file', type=str, default=os.path.join('results_all', 'serials', 'serial_coverage.csv'))
    args = parser.parse_args()

    w = pd.read_csv(args.wandb_runs, low_memory=False)
    out = w.groupby('serial').agg(wandb_runs=('run_id', 'size'), finished=('state', lambda s: int((s == 'finished').sum())),
                                  first_run=('created_at', 'min'), last_run=('created_at', 'max'))
    out[['first_run', 'last_run']] = out[['first_run', 'last_run']].apply(lambda c: c.str[:10])

    counts = csv_rows(args.data_dir)
    out['csv_rows'] = [counts.get(s, (0, ''))[0] for s in out.index]
    out['csv_file'] = [counts.get(s, (0, ''))[1] for s in out.index]

    # a `serialNN_ckpts` folder names the wandb serial; files renamed into one keep their old serial inside
    inv = pd.read_csv(os.path.join(args.checkpoints, 'checkpoint_inventory.csv'))
    inv['paper_serial'] = pd.to_numeric(inv.folder.str.extract(r'serial(\d+)_ckpts$')[0]).fillna(inv.serial)
    out['checkpoints'] = inv.groupby('paper_serial').size().reindex(out.index).fillna(0).astype(int)

    doc = documented_serials(args.serials_doc)
    out['in_serials_md'] = [s in doc for s in out.index]
    missing = sorted(s for s in doc if s not in out.index)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out.to_csv(args.output_file)
    pd.set_option('display.width', 200, 'display.max_rows', 200)
    print(out.to_string())
    print(f'{len(w)} runs in {len(out)} serials; undocumented: {sorted(out.index[~out.in_serials_md].tolist())}; documented but not on wandb: {missing}')


if __name__ == '__main__':
    main()
