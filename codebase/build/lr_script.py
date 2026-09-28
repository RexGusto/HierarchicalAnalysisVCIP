import os
import argparse
import pandas as pd


MODEL_TAGS = {
    # ResNet
    "hiresnet50.tv_in1k": "tv1",
    "hiresnet50.gluon_in1k": "gluon",
    "hiresnet50.in21k_miil": "rn_miil",
    "hiresnet50.a1_in1k": "a1",
    "hiresnet50.tv2_in1k": "tv2",

    "hiresnet50.fb_swsl_ig1b_ft_in1k": "ig1b",
    "hiresnet50.fb_ssl_yfcc100m_ft_in1k": "yfcc100m",

    "hiresnet50.in1k_supcon": "supcon",
    "hiresnet50.in1k_swav": "swav",
    "hiresnet50.in1k_mocov3": "rn_mocov3",
    "hiresnet50.in1k_spark": "spark",

    # ViT
    "hivit_base_patch16_224.orig_in21k": "vit_b",
    "hideit_base_patch16_224.fb_in1k": "deit",
    "hivit_base_patch16_224_miil.in21k": "vit_miil",
    "hideit3_base_patch16_224.fb_in1k": "deit3",
    "hideit3_base_patch16_224.fb_in22k_ft_in1k": "deit3in21k",

    "hivit_base_patch16_224.in1k_mocov3": "vit_mocov3",
    "hivit_base_patch16_224.dino": "dino",
    "hivit_base_patch16_clip_224.laion2b": "clip",
    "hivit_base_patch16_siglip_224.v2_webli": "siglipv2",
    "hivit_base_patch16_224.mae": "mae",
}

def get_best_lr_opt_wd(
    df_subset, dataset, method,
    selection_var='val_acc', lr_var='lr', train_acc_th=50
    ):

    # sort subset based on val loss (lowest to highest)
    ascending = True if 'loss' in selection_var else False
    df_subset_sorted = df_subset.sort_values(by=[selection_var], ascending=ascending)

    # get index and train acc corresponding to best val loss (0)
    best_idx = 0
    train_acc = df_subset_sorted['train_acc_level1'].iloc[best_idx]

    while train_acc < train_acc_th and best_idx < (len(df_subset) - 1):
        # increment index of next best val loss
        best_idx += 1
        # get val loss and train acc corresponding to next best
        train_acc = df_subset_sorted['train_acc_level1'].iloc[best_idx]
        # print when train_acc < 50 (mostly cotton/soy/ultra fine-grained datasets)
        print(f'Acc below {train_acc_th}: {dataset}, {method}, {best_idx}, {train_acc}')
    
    # if none of the accuracies surpass TH then just return the highest accuracies
    if train_acc < train_acc_th:
        # sort subset based on train acc (highest to lowest)
        df_subset_sorted = df_subset.sort_values(by=['train_acc_level1'], ascending=False) 
        print(f'All below train acc threshold ({train_acc_th}): {dataset}, {method}')
        print(df_subset_sorted)
        lr = df_subset_sorted[lr_var].iloc[0]
        opt = df_subset_sorted['opt'].iloc[0]
        wd = df_subset_sorted['weight_decay'].iloc[0]

    # returns lr corresponding to best val loss
    else:
        lr = df_subset_sorted[lr_var].iloc[best_idx]
        opt = df_subset_sorted['opt'].iloc[best_idx]
        wd = df_subset_sorted['weight_decay'].iloc[best_idx]
    
    # check the current subset and the selected LR
    # print(dataset, method, lr, best_idx, df_subset.head())

    return lr, opt, wd


def get_lr_cmd(
    df_subset, dataset, method, suffix, prefix,
    selection_var='val_acc', lr_var='lr', train_acc_th=50,
    cluster_ratio=None
    ):
    lr, opt, wd = get_best_lr_opt_wd(
        df_subset, dataset, method, selection_var, lr_var, train_acc_th
    )

    #write to file
    model_name = df_subset['model_name'].iloc[0]
    model_tag = MODEL_TAGS.get(model_name, "unknown")

    if cluster_ratio is not None:
        cfg_path = f"datasets/{dataset}/{dataset}_pseudo_{model_tag}_{int(cluster_ratio)}.yaml --use_hierarchy --ignore_pl_eval"
    else:
        cfg_path = f"datasets/{dataset}.yaml"

    fz = ' --freeze_backbone' if 'fz' in method else ''

    classifier = df_subset['classifier'].iloc[0]
    classifier = f" --classifier {classifier.replace('_', '', 1)}" if classifier else ''

    adapter = df_subset['adapter'].iloc[0]
    adapter = f" --adapter {adapter.replace('_', '', 1)}" if adapter else ''

    prompt = df_subset['prompt'].iloc[0].replace('_', '', 1)
    prompt = f" --prompt {prompt}" if prompt else ''

    opt_text = f' --opt {opt} --weight_decay {wd}'
    method_text = f'{fz}{adapter}{prompt}{opt_text}'
    others = f'{method_text}{suffix}'

    line = f'{prefix} --cfg configs/{cfg_path} --{lr_var} {lr} --model_name {model_name}{others}\n'
    return line


def make_lr_script(args):
    df = pd.read_csv(args.input_file)
    cols = [
        'serial', 'dataset_name', 'model_name', 'freeze_backbone',
        'classifier', 'adapter', 'prompt',
        args.selection_var, 'train_acc_level1',
        'lr', 'base_lr', 'opt', 'weight_decay'
    ]

    if 'n_cluster_ratio' in df.columns:
        cols.append('n_cluster_ratio') 

    df = df[cols]

    # dataset and method names
    dataset_list = df['dataset_name'].unique()

    df = df.fillna({'classifier': '', 'adapter': '', 'prompt': ''})

    df['freeze_backbone'] = df['freeze_backbone'].apply(lambda x: '_fz' if x is True else '')
    df['classifier'] = df['classifier'].apply(lambda x: f'_{x}' if x else '')
    df['adapter'] = df['adapter'].apply(lambda x: f'_{x}' if x else '')
    df['prompt'] = df['prompt'].apply(lambda x: f'_{x}' if x else '')

    df['method'] = df['model_name'] + df['classifier'] + df['adapter'] + df['prompt'] + df['freeze_backbone']

    # create file with specified file name
    output_file = os.path.join(args.results_dir, f'{args.output_file}.sh')
    f = open(output_file, "w")


    #for loop for each dataset
    for dataset in dataset_list:
        # write dataset name
        f.write(f'# {dataset}\n')

        prefix = args.prefix
        lr_var = args.lr_var
        suffix = args.suffix

        # for loop for each method
        method_list = df[df['dataset_name'] == dataset]['method'].unique()

        df_dataset = df[df['dataset_name'] == dataset]

        if args.use_ratios:
            # get all ratios for this dataset
            ratios = df_dataset['n_cluster_ratio'].dropna().unique()
            ratios = sorted(ratios)

            for ratio in ratios:
                f.write(f'# {dataset}, ratio: {int(ratio)}\n')

                df_ratio = df_dataset[df_dataset['n_cluster_ratio'] == ratio]

                method_list = df_ratio['method'].unique()

                for method in method_list:
                    df_subset = df_ratio[df_ratio['method'] == method].copy(deep=False)
                    df_subset.dropna(subset=args.selection_var, inplace=True)

                    if len(df_subset) == 0:
                        print(dataset, method, ratio)
                        continue

                    line = get_lr_cmd(
                        df_subset, dataset, method, suffix, prefix,
                        args.selection_var, lr_var, args.train_acc_th,
                        cluster_ratio=ratio
                    )

                    line = line.strip() + f' --n_cluster_ratio {int(ratio)}\n'
                    f.write(line)

                f.write('\n')

        else:
            # original behavior (no ratios)
            method_list = df_dataset['method'].unique()

            f.write(f'# {dataset}\n')

            prev_serial = None

            for method in method_list:
                df_subset = df_dataset[df_dataset['method'] == method].copy(deep=False)
                df_subset.dropna(subset=args.selection_var, inplace=True)

                if len(df_subset) == 0:
                    print(dataset, method)
                    continue

                serial = df_subset['serial'].iloc[0]

                if serial != prev_serial:
                    f.write(f'\n# serial = {serial}\n')
                    prev_serial = serial


                line = get_lr_cmd(
                    df_subset, dataset, method, suffix, prefix,
                    args.selection_var, lr_var, args.train_acc_th,
                    cluster_ratio=None
                )

                f.write(line)

            f.write('\n')

    f.close()

    return 0


def parse_args():

    parser = argparse.ArgumentParser()

    #parser arguments
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('data', 'hierarchical_soylocal.csv'),
                        help='filename for input .csv file')

    parser.add_argument('--selection_var', type=str, default='val_acc_level1')
    parser.add_argument('--lr_var', type=str, default='lr',
                        help='for inat/dafb use --lr_var base_lr')
    parser.add_argument('--train_acc_th', type=int, default=20)
    parser.add_argument('--prefix', type=str,
                        default='python -u tools/train.py --serial 56',
                        help='prefix for the file in each line')
    parser.add_argument('--suffix', type=str,
                        default=' --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448',
                        help='suffix for the file in each line')
    parser.add_argument('--output_file', type=str, default='hierarchical_ft_stage2',
                        help='output file name')
    parser.add_argument('--use_ratios', action='store_true',
                        help='flag when finding best lr per ratios')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'lr_scripts'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    make_lr_script(args)

    return 0


if __name__ == '__main__':
    main()

