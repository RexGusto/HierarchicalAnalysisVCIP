import os
import shlex
import argparse
import pandas as pd


# machine-specific flags; the rest of the recorded launch args is kept verbatim
DROP = {'--results_dir', '--cpu_workers'}
WEIGHT_DECAY = {'sgd': '0', 'adamw': '5e-2'}


def parse(args):
    tokens, out, i = shlex.split(args), [], 0
    while i < len(tokens):
        flag = tokens[i]
        has_value = i + 1 < len(tokens) and not tokens[i + 1].startswith('--')
        value = tokens[i + 1] if has_value else None
        out.append((flag, value))
        i += 2 if has_value else 1
    return out


def render(pairs, serial, overrides=None):
    overrides = overrides or {}
    seen, parts = set(), []
    for flag, value in pairs:
        if flag in DROP:
            continue
        value = str(serial) if flag == '--serial' else overrides.get(flag, value)
        seen.add(flag)
        parts.append(flag if value is None else f'{flag} {value}')
    parts += [f'{k} {v}' for k, v in overrides.items() if k not in seen]
    return 'python -u tools/train.py ' + ' '.join(parts)


def commands_from_runs(runs, serial):
    runs = runs.assign(finished=runs.state == 'finished').sort_values(['finished', 'created_at'], ascending=False)
    lines = {}
    for args in runs.args:
        pairs = parse(args)
        key = render([(f, v) for f, v in pairs if f != '--serial'], serial)
        lines.setdefault(key, render(pairs, serial))
    return sorted(lines.values())


def gap_commands(index, serial, gap_from):
    # configurations of `gap_from` serials with no finished run in `serial`, built from a finished run
    # of the same dataset and model in `serial`
    idx = index.assign(lr=index.lr.astype(float).round(10))
    key = ['dataset_name', 'model_name', 'opt', 'lr']
    wanted = idx[idx.serial.isin([serial] + gap_from)][key].drop_duplicates()
    done = idx[(idx.serial == serial) & (idx.state == 'finished')][key].drop_duplicates()
    missing = wanted.merge(done, on=key, how='left', indicator=True).query('_merge == "left_only"')
    return missing[key]


def main():
    parser = argparse.ArgumentParser(description='Write rerun_<serial>.sh: one literal train command per configuration, from recorded wandb launch args')
    parser.add_argument('--serial', type=int, required=True)
    parser.add_argument('--run_args', type=str, required=True, help='data/run_args/serial_<N>.csv')
    parser.add_argument('--gaps_only_from', nargs='*', type=int, default=None,
                        help='only write configurations missing a finished run, taken from these serials and --serial')
    parser.add_argument('--wandb_runs', type=str, default=os.path.join('data', 'wandb_runs.csv'))
    parser.add_argument('--commit', type=str, required=True, help='code commit the reruns must use')
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--output_dir', type=str, default=os.path.join('codebase', 'experiments'))
    args = parser.parse_args()

    runs = pd.read_csv(args.run_args)
    if args.gaps_only_from is None:
        lines = commands_from_runs(runs, args.serial)
    else:
        missing = gap_commands(pd.read_csv(args.wandb_runs, low_memory=False), args.serial, args.gaps_only_from)
        runs = runs[runs.state == 'finished'].assign(lr=runs.lr.astype(float).round(10))
        lines = []
        for _, m in missing.iterrows():
            template = runs[(runs.dataset_name == m.dataset_name) & (runs.model_name == m.model_name)]
            if template.empty:
                raise ValueError(f'no finished serial-{args.serial} run to template {m.to_dict()}')
            lr = f'{m.lr:g}'
            lines.append(render(parse(template.args.iloc[0]), args.serial,
                                {'--lr': lr, '--opt': m.opt, '--weight_decay': WEIGHT_DECAY[m.opt]}))
        lines = sorted(set(lines))

    os.makedirs(args.output_dir, exist_ok=True)
    fp = os.path.join(args.output_dir, f'rerun_{args.serial}.sh')
    header = [f'# Reruns of serial {args.serial}: {len(lines)} runs. Run from projects/Hierarchical at commit {args.commit} or later.',
              f'# Before launching, move the old serial-{args.serial} runs to the wandb project nycu_pcs/Others.'
              if args.gaps_only_from is None else
              f'# Adds the missing runs to serial {args.serial}; its finished runs stay. Its unfinished runs go to nycu_pcs/Others.']
    if args.note:
        header.append(f'# {args.note}')
    with open(fp, 'w') as f:
        f.write('\n'.join(header + lines) + '\n')
    print(f'{len(lines)} commands -> {fp}')


if __name__ == '__main__':
    main()
