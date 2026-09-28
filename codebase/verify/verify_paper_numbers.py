import os
import re
import sys
import argparse
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from PIL import Image


DEIT = 'hideit_base_patch16_224.fb_in1k'
VIT = 'hivit_base_patch16_224.orig_in21k'
SPARK = 'hiresnet50.in1k_spark'
SWAV = 'hiresnet50.in1k_swav'
PAPER_FIGURES = ['compare_soylocal', 'compare_resnet_soylocal', 'compare_vit_soylocal']


def rnd(x, dp):
    return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def printed(tex, pattern):
    # the printed value is the pattern's first group; its decimals set the rounding
    m = re.search(pattern, tex, flags=re.S)
    if m is None:
        raise ValueError(f'pattern not found in the paper: {pattern}')
    s = m.group(1)
    return float(s), len(s.split('.')[1]) if '.' in s else 0


def load(data_dir):
    soy = pd.read_csv(os.path.join(data_dir, 'hierarchical_soylocal.csv'))
    soy = soy[soy['dataset_name'] == 'soylocal']
    leaves = pd.read_csv(os.path.join(data_dir, 'other_soy_stuff.csv'))
    hph = soy[soy['serial'] == 54].groupby('model_name')['val_acc_level1'].max()
    base = soy[soy['serial'] == 55].groupby('model_name')['val_acc_level1'].max()
    assert len(hph) == len(base) == 21, (len(hph), len(base))
    return soy, leaves, hph, base


def number_checks(tex, soy, leaves, hph, base):
    resnets = [m for m in hph.index if 'resnet' in m]
    vits = [m for m in hph.index if 'resnet' not in m]
    table = lambda row: re.search(row + r'[^&]*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)', tex).groups()
    csdnet = [float(v) for v in table(r'CSDNet\\cite\{[^}]*\}')]

    def leaf_max(dataset, model):
        g = leaves[(leaves['serial'] == 86) & (leaves['dataset_name'] == dataset) & (leaves['model_name'] == model)]
        return g['val_acc_level1'].max()

    rows = []
    def check(label, pattern, derive):
        value, dp = printed(tex, pattern)
        rows.append(dict(number=label, paper=value, derived=rnd(derive(dp), dp)))

    check('DeiT HPH max on SoyLocal', r'maximum accuracy of \\textbf\{([\d.]+)', lambda dp: hph[DEIT])
    # the paper does not print the two maxima, so a gain is the stored difference rounded once
    check('DeiT gain over baseline', r'DeiT obtains the largest gain of \\textbf\{([\d.]+)', lambda dp: hph[DEIT] - base[DEIT])
    check('SparK gain over baseline', r'ResNet SparK achieves the largest improvement, increasing the maximum accuracy by \\textbf\{([\d.]+)', lambda dp: hph[SPARK] - base[SPARK])
    check('largest gain over baseline (abstract)', r'absolute gains of up to \\textbf\{([\d.]+)', lambda dp: (hph - base).max())
    check('table: ViT HPH SoyLocal', r'HPH \(Ours\) & \\underline\{[\d.]+\} & \\underline\{[\d.]+\} & \\underline\{([\d.]+)\}', lambda dp: hph[VIT])
    check('table: ViT HPH Cotton', r'HPH \(Ours\) & \\underline\{([\d.]+)\}', lambda dp: leaf_max('cotton', VIT))
    check('table: ViT HPH SoyAgeing-R1', r'HPH \(Ours\) & \\underline\{[\d.]+\} & \\underline\{([\d.]+)\}', lambda dp: leaf_max('soyageing_r1', VIT))
    check('table: DeiT HPH Cotton', r'HPH \(Ours\)\s*& \\textbf\{([\d.]+)\}', lambda dp: leaf_max('cotton', DEIT))
    check('table: DeiT HPH SoyAgeing-R1', r'HPH \(Ours\)\s*& \\textbf\{[\d.]+\} & \\textbf\{([\d.]+)\}', lambda dp: leaf_max('soyageing_r1', DEIT))
    check('table: DeiT HPH SoyLocal', r'HPH \(Ours\)\s*& \\textbf\{[\d.]+\} & \\textbf\{[\d.]+\} & \\textbf\{([\d.]+)\}', lambda dp: hph[DEIT])
    check('gain over CSDNet, SoyLocal', r'improves accuracy by ([\d.]+)\\% on SoyLocal', lambda dp: rnd(hph[DEIT], 1) - csdnet[2])
    check('gain over CSDNet, Cotton', r'([\d.]+)\\% on Cotton', lambda dp: rnd(leaf_max('cotton', DEIT), 1) - csdnet[0])
    check('gain over CSDNet, SoyAgeing-R1', r'and ([\d.]+)\\% on SoyAgeing-R1', lambda dp: rnd(leaf_max('soyageing_r1', DEIT), 1) - csdnet[1])

    params = soy[soy['serial'] == 54].groupby(['model_name', 'n_cluster_ratio'])['no_params_trainable'].first().unstack()
    # the paper does not name the backbone: this passes if any backbone has the printed count
    value, dp = printed(tex, r'contains ([\d.]+)M trainable parameters')
    matching = params.index[params[10].round(dp) == value]
    rows.append(dict(number=f"trainable params at 10% ratio (M), low specificity: {', '.join(matching) or 'no backbone'}",
                     paper=value, derived=value if len(matching) else rnd(params.loc[DEIT, 10], dp)))
    model = matching[0] if len(matching) else DEIT
    check('params per 10% ratio step (M)', r'introduces approximately ([\d.]+)M', lambda dp: (params.loc[model, 90] - params.loc[model, 10]) / 8)
    check('params 10% to 90% (M)', r'increases the trainable parameter count by approximately ([\d.]+)M', lambda dp: params.loc[model, 90] - params.loc[model, 10])
    check('SoyLocal training images', r'SoyLocal contains \d+ classes with ([\d]+) training images', lambda dp: soy['num_images_train'].max())
    check('Cotton training images', r'Cotton contains \d+ classes with ([\d]+) training images', lambda dp: leaves.loc[leaves['dataset_name'] == 'cotton', 'num_images_train'].max())
    check('SoyAgeing-R1 training images', r'SoyAgeing-R1 contains \d+ classes with ([\d]+) training images', lambda dp: leaves.loc[(leaves['serial'] == 86) & (leaves['dataset_name'] == 'soyageing_r1'), 'num_images_train'].min())
    check('backbones', r'using ([\d]+) pretrained backbones \(', lambda dp: len(hph))
    check('CNN backbones', r'backbones \(([\d]+) CNNs', lambda dp: len(resnets))
    check('ViT backbones', r'CNNs and ([\d]+) ViTs\)', lambda dp: len(vits))

    gains = hph - base
    per_ratio = soy[soy['serial'] == 54].groupby('n_cluster_ratio')['val_acc_level1'].mean() - soy[soy['serial'] == 55]['val_acc_level1'].mean()
    r1 = leaves[(leaves['serial'] == 86) & (leaves['dataset_name'] == 'soyageing_r1')]
    r1_best = r1.loc[r1.groupby('model_name')['val_acc_level1'].idxmax(), 'num_images_train']
    claims = [
        ('HPH mean accuracy beats the baseline at every cluster ratio', bool((per_ratio > 0).all())),
        ('HPH raises the maximum accuracy of every backbone', bool((gains > 0).all())),
        ('SparK has the largest gain among ResNets', gains[resnets].idxmax() == SPARK),
        ('DeiT has the largest gain among ViTs', gains[vits].idxmax() == DEIT),
        ('DeiT has the highest HPH maximum overall', hph.idxmax() == DEIT),
        ('SwAV is the only ResNet at or above 60% with HPH', list(hph[resnets][hph[resnets] >= 60].index) == [SWAV]),
        ('SoyAgeing-R1 table values use the 5-image-per-class split', bool((r1_best < 1000).all())),
    ]
    return pd.DataFrame(rows), claims, gains


def figure_checks(plots_dir, paper_dir):
    rows = []
    for name in PAPER_FIGURES:
        new = np.asarray(Image.open(os.path.join(plots_dir, f'{name}.png')).convert('RGB'), dtype=np.int16)
        old = np.asarray(Image.open(os.path.join(paper_dir, f'{name}.png')).convert('RGB'), dtype=np.int16)
        # tight bounding boxes can round a pixel differently; compare the best alignment within 2 px
        close = all(abs(a - b) <= 2 for a, b in zip(new.shape, old.shape))
        h, w = min(new.shape[0], old.shape[0]) - 2, min(new.shape[1], old.shape[1]) - 2
        differ = min(float((np.abs(new[dy:dy + h, dx:dx + w] - old[:h, :w]).max(axis=2) > 32).mean())
                     for dy in range(3) for dx in range(3)) if close else float('nan')
        rows.append(dict(figure=name, size_new='x'.join(map(str, new.shape[1::-1])), size_paper='x'.join(map(str, old.shape[1::-1])),
                         same_size=close, pixels_differing=round(differ, 4)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Re-derive every data-backed number and claim printed in the VCIP26 paper')
    parser.add_argument('--tex', type=str, default=os.path.join('drafts', 'VCIP26', 'latex', 'camera_ready.tex'))
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--plots_dir', type=str, default=os.path.join('results_all', 'plots'))
    # rendering differences across matplotlib versions reach ~1.1%; the 54/55 variant of compare_soylocal differs by 1.7%
    parser.add_argument('--max_pixels_differing', type=float, default=0.015)
    args = parser.parse_args()

    with open(args.tex) as f:
        tex = f.read()
    numbers, claims, gains = number_checks(tex, *load(args.data_dir))
    numbers['ok'] = (numbers['paper'] - numbers['derived']).abs() < 1e-9
    figures = figure_checks(args.plots_dir, os.path.join(os.path.dirname(args.tex), 'figures'))
    figures['ok'] = figures['same_size'] & (figures['pixels_differing'] <= args.max_pixels_differing)

    pd.set_option('display.width', 200, 'display.max_colwidth', 70)
    print(numbers.to_string(index=False))
    print()
    for claim, holds in claims:
        print(f"{'ok  ' if holds else 'FAIL'} {claim}")
    print()
    print('HPH minus baseline maximum accuracy per backbone (SoyLocal):')
    print(gains.sort_values().round(2).to_string())
    print()
    print(figures.to_string(index=False))

    failed = int((~numbers['ok']).sum()) + sum(not h for _, h in claims) + int((~figures['ok']).sum())
    total = len(numbers) + len(claims) + len(figures)
    print(f'\n{total - failed}/{total} checks pass')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
