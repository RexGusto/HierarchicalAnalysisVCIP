import io
import argparse
import subprocess

import pandas as pd


def git_lines(*cmd):
    return subprocess.run(['git', *cmd], capture_output=True, text=True, check=True).stdout.splitlines()


def compare(path, ref):
    new = pd.read_csv(path)
    old = pd.read_csv(io.StringIO('\n'.join(git_lines('show', f'{ref}:{path}'))))
    shared = [c for c in old.columns if c in new.columns]
    numeric = [c for c in shared if pd.api.types.is_numeric_dtype(old[c]) and c not in ('serial', 'seed', 'n_cluster_ratio', 'lr', 'lr_acc_max')]
    keys = [c for c in shared if c not in numeric]
    for df in (old, new):
        for k in keys:
            # 50 and 50.0 are the same key
            as_num = pd.to_numeric(df[k], errors='coerce')
            df[k] = as_num.astype(float).round(10).astype(str).where(as_num.notna() | df[k].isna(), df[k].astype(str))
    merged = old[shared].merge(new[shared], on=keys, how='outer', suffixes=('_old', '_new'), indicator=True)
    unmatched = int((merged['_merge'] != 'both').sum())
    both = merged[merged['_merge'] == 'both']
    diffs = {c: (both[f'{c}_old'] - both[f'{c}_new']).abs().max() for c in numeric}
    max_diff = max([d for d in diffs.values() if pd.notna(d)], default=0.0)
    return dict(file=path, rows_old=len(old), rows_new=len(new), matched=len(both), unmatched=unmatched,
                max_abs_diff=round(float(max_diff), 6), cols_added=len(set(new.columns) - set(old.columns)),
                cols_removed=len(set(old.columns) - set(new.columns)))


def main():
    parser = argparse.ArgumentParser(description='Compare regenerated CSVs under a path to their committed version by value, ignoring column order and added columns')
    parser.add_argument('--path', type=str, default='results_all')
    parser.add_argument('--ref', type=str, default='HEAD')
    args = parser.parse_args()

    changed = [f for f in git_lines('diff', '--name-only', args.ref, '--', args.path) if f.endswith('.csv')]
    rows = [compare(f, args.ref) for f in changed]
    pd.set_option('display.width', 250, 'display.max_colwidth', 90, 'display.max_rows', 500)
    print(pd.DataFrame(rows).to_string(index=False) if rows else f'no CSV under {args.path} differs from {args.ref}')


if __name__ == '__main__':
    main()
