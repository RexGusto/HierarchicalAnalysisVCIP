import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon


METRICS = ['max', 'mean', 'cv', 'regret', 'basin2', 'collapse']
HIGHER_IS_BETTER = {'max', 'mean', 'basin2'}


def load_pairs(real_csv, soylocal_csv):
    # hierarchical vs baseline: 23 vs 32 on aircraft/cub/cars, 54 vs 55 on SoyLocal
    real = pd.read_csv(real_csv, low_memory=False)
    real = real[real.serial.isin([23, 32])].assign(
        dataset=lambda d: d.dataset_name.str.replace('_baseline', ''),
        model=lambda d: d.model_name.str.replace('^(?!hi)', 'hi', regex=True),
        setting=lambda d: d.serial.map({23: 'hier', 32: 'base'}), n_cluster_ratio=0)
    soy = pd.read_csv(soylocal_csv, low_memory=False).assign(
        dataset=lambda d: d.dataset_name, model=lambda d: d.model_name,
        setting=lambda d: d.serial.map({54: 'hier', 55: 'base'}))
    soy['n_cluster_ratio'] = soy.n_cluster_ratio.fillna(0)
    runs = pd.concat([real, soy]).dropna(subset=['val_acc_level1'])
    return runs.drop_duplicates(subset=['serial', 'dataset', 'model', 'opt', 'lr', 'n_cluster_ratio'])


def sweep_stats(acc):
    best = acc.max()
    return pd.Series(dict(n=len(acc), max=best, mean=acc.mean(), cv=100 * acc.std(ddof=0) / acc.mean(),
                          regret=(best - acc).mean(), basin2=(acc >= best - 2).mean(), collapse=(acc < 0.5 * best).mean()))


def hier_vs_base(runs):
    rows, per_dataset = [], []
    for opt in ['sgd', 'adamw', 'both']:
        sel = runs if opt == 'both' else runs[runs.opt == opt]
        # one sweep per (backbone, dataset, setting, ratio); SoyLocal ratios are averaged
        stats = sel.groupby(['dataset', 'model', 'setting', 'n_cluster_ratio']).val_acc_level1.apply(sweep_stats).unstack()
        stats = stats.groupby(['dataset', 'model', 'setting']).mean().unstack('setting')
        for m in METRICS:
            diff = (stats[(m, 'hier')] - stats[(m, 'base')]).dropna()
            better = diff > 0 if m in HIGHER_IS_BETTER else diff < 0
            rows.append(dict(opt=opt, metric=m, pairs=len(diff), mean_diff=diff.mean(),
                             frac_hier_better=better.mean(), wilcoxon_p=wilcoxon(diff).pvalue))
            for ds, d in diff.groupby(level='dataset'):
                per_dataset.append(dict(opt=opt, metric=m, dataset=ds, pairs=len(d), mean_diff=d.mean()))
    return pd.DataFrame(rows), pd.DataFrame(per_dataset)


def ranking_stability(stage1_csv):
    st = pd.read_csv(stage1_csv, low_memory=False)
    st = st[st.serial == 23].drop_duplicates(['dataset_name', 'model_name', 'opt', 'lr'])
    by_max = st.groupby(['model_name', 'dataset_name']).val_acc_level1.max().unstack()
    by_mean = st.groupby(['model_name', 'dataset_name']).val_acc_level1.mean().unstack()
    rows = []
    for i, a in enumerate(by_max.columns):
        for b in by_max.columns[i + 1:]:
            rows.append(dict(pair=f'{a} vs {b}', tau_by_max=kendalltau(by_max[a], by_max[b])[0],
                             tau_by_mean=kendalltau(by_mean[a], by_mean[b])[0]))
    for ds in by_max.columns:
        top = by_max[ds].sort_values(ascending=False)
        rows.append(dict(pair=f'{ds}: max vs mean', tau_by_max=kendalltau(by_max[ds], by_mean[ds])[0],
                         best=top.index[0], gap_best_second=top.iloc[0] - top.iloc[1]))
    return pd.DataFrame(rows)


def seed_noise(seeds_csv):
    cs = pd.read_csv(seeds_csv, low_memory=False)
    sd = cs.groupby(['serial', 'dataset_name', 'model_name', 'n_cluster_ratio', 'opt', 'lr'], dropna=False).val_acc_level1.agg(['size', 'std'])
    sd = sd[sd['size'] >= 3]['std']
    return pd.DataFrame([dict(configs=len(sd), median_std=sd.median(), mean_std=sd.mean(), max_std=sd.max())])


def family_by_optimizer(stage1_csv):
    st = pd.read_csv(stage1_csv, low_memory=False)
    st = st[st.serial == 23].drop_duplicates(['dataset_name', 'model_name', 'opt', 'lr'])
    st['family'] = np.where(st.model_name.str.contains('resnet'), 'ResNet', 'ViT')
    stats = st.groupby(['dataset_name', 'model_name', 'family', 'opt']).val_acc_level1.apply(sweep_stats).unstack()
    return stats.groupby(['family', 'opt'])[['regret', 'cv', 'max']].mean().reset_index()


def main():
    parser = argparse.ArgumentParser(description='Hierarchical vs baseline robustness per optimizer, ranking stability, seed noise')
    parser.add_argument('--real_csv', default=os.path.join('data', 'hierarchical_real.csv'))
    parser.add_argument('--soylocal_csv', default=os.path.join('data', 'hierarchical_soylocal.csv'))
    parser.add_argument('--stage1_csv', default=os.path.join('data', 'hierarchical_stage1.csv'))
    parser.add_argument('--seeds_csv', default=os.path.join('data', 'cotton_seeds.csv'))
    parser.add_argument('--results_dir', default=os.path.join('results_all', 'robustness'))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    overall, per_dataset = hier_vs_base(load_pairs(args.real_csv, args.soylocal_csv))
    tables = dict(hier_vs_base=overall, hier_vs_base_per_dataset=per_dataset,
                  ranking_stability=ranking_stability(args.stage1_csv), seed_noise=seed_noise(args.seeds_csv),
                  family_by_optimizer=family_by_optimizer(args.stage1_csv))
    pd.set_option('display.width', 200)
    for name, df in tables.items():
        df.to_csv(os.path.join(args.results_dir, f'{name}.csv'), index=False)
        print(f'== {name}'); print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
