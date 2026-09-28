import numpy as np
import pandas as pd


SERIALS_EXPLANATIONS = [
    # best lr acc
    'ft_448',

    # main experiments
    'ft_448',
    'fz_448',
]


SETTINGS_DIC = {
    'ft_448': 'FT',
    'fz_448': 'FZ',
}

SERIAL_DIC = {
    32: 'Baseline (BL)',
    23: 'Real Hierarchy (RH)',
    24: 'FZ',
    39: 'Pseudo-Hierarchy',
    40: 'Pseudo-Hierarchy (PH)',
    51: 'Different Extractors',
    52: 'Kmeans',
    53: 'Agglomerative',
    62: 'Baseline (bs)',
    63: 'Pseudo-Hierarchy (bs)',
    64: 'Baseline (epochs)',
    65: 'Pseudo-Hierarchy (epochs)',
    66: 'Baseline (is)',
    67: 'Pseudo-Hierarchy (is)',
    68: 'Baseline (augs)',
    69: 'Pseudo-Hierarchy (augs)',
    55: 'Baseline', # Soylocal
    54: 'Pseudo-Hierarchy', # Soylocal
    83: 'BL', # Other leaves
    86: 'PH', # Other leaves
    90: 'PH + HLS',
    91: 'PH + SCM1',
    92: 'PH + HLS + SCM2',
    93: 'PH + HLS + SCM1',
    94: 'BL + LS',
    95: 'BL + CM',
    102: '1-example baseline',
    103: '1-example hierarchy'
}

METHODS_RESNET = [
    'hiresnet50.tv_in1k',
    'hiresnet50.gluon_in1k',
    'hiresnet50.in21k_miil',
    'hiresnet50.a1_in1k',
    'hiresnet50.tv2_in1k',

    'hiresnet50.fb_swsl_ig1b_ft_in1k',
    'hiresnet50.fb_ssl_yfcc100m_ft_in1k',

    'hiresnet50.in1k_supcon',
    'hiresnet50.in1k_swav',
    'hiresnet50.in1k_mocov3',
    'hiresnet50.in1k_spark',

    'hiresnet50.tv_in1k_fz',
    'hiresnet50.gluon_in1k_fz', 
    'hiresnet50.in21k_miil_fz',
    'hiresnet50.a1_in1k_fz',
    'hiresnet50.tv2_in1k_fz',

    'hiresnet50.fb_swsl_ig1b_ft_in1k_fz',
    'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz',

    'hiresnet50.in1k_supcon_fz',
    'hiresnet50.in1k_swav_fz',
    'hiresnet50.in1k_mocov3_fz',
    'hiresnet50.in1k_spark_fz', 
]

METHODS_VIT = [
    'hivit_base_patch16_224.orig_in21k',
    'hideit_base_patch16_224.fb_in1k',
    'hivit_base_patch16_224_miil.in21k',
    'hideit3_base_patch16_224.fb_in1k',
    'hideit3_base_patch16_224.fb_in22k_ft_in1k',

    'hivit_base_patch16_224.in1k_mocov3',
    'hivit_base_patch16_224.dino',
    'hivit_base_patch16_clip_224.laion2b',
    'hivit_base_patch16_siglip_224.v2_webli',
    'hivit_base_patch16_224.mae', 

    'hivit_base_patch16_224.orig_in21k_fz',
    'hideit_base_patch16_224.fb_in1k_fz',
    'hivit_base_patch16_224_miil.in21k_fz',
    'hideit3_base_patch16_224.fb_in1k_fz',
    'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz',

    'hivit_base_patch16_224.in1k_mocov3_fz',
    'hivit_base_patch16_224.dino_fz',
    'hivit_base_patch16_clip_224.laion2b_fz',
    'hivit_base_patch16_siglip_224.v2_webli_fz',
    'hivit_base_patch16_224.mae_fz', 
]


DATASETS_DIC = {
    'aircraft': 'Aircraft',
    'cars': 'Cars',
    'cotton': 'Cotton',
    'cub': 'CUB',
    'dogs': 'Dogs',
    'dafb': 'DAFB',
    'flowers': 'Flowers',
    'food': 'Food',
    'inat17': 'iNat17',
    'moe': 'Moe',
    'nabirds': 'NABirds',
    'pets': 'Pets',
    'soyageing': 'SoyAgeing',
    'soyageingr1': 'SoyAgeingR1',
    'soyageingr3': 'SoyAgeingR3',
    'soyageingr4': 'SoyAgeingR4',
    'soyageingr5': 'SoyAgeingR5',
    'soyageingr6': 'SoyAgeingR6',
    'soygene': 'SoyGene',
    'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal',
    'vegfru': 'VegFru',
    'all': 'Average',
}


METHODS_DIC = {
    # ResNet FSL models
    'hiresnet50.tv_in1k': 'RN TV1',
    'hiresnet50.tv2_in1k': 'RN TV2',
    'hiresnet50.gluon_in1k': 'RN Gluon',
    'hiresnet50.in21k_miil': 'RN IN21k-P',
    'hiresnet50.a1_in1k': 'RN A1',

    # ResNet Semi-SL models
    'hiresnet50.fb_swsl_ig1b_ft_in1k': 'RN IG1b',
    'hiresnet50.fb_ssl_yfcc100m_ft_in1k': 'RN YFCC100m',

    # ResNet SSL models
    'hiresnet50.in1k_supcon': 'RN SupCon',
    'hiresnet50.in1k_swav': 'RN SwAV',
    'hiresnet50.in1k_mocov3': 'RN MoCo v3',
    'hiresnet50.in1k_spark': 'RN SparK',

    # ResNet FSL Frozen models 
    'hiresnet50.tv_in1k_fz': 'RN TV1',
    'hiresnet50.gluon_in1k_fz': 'RN Gluon',
    'hiresnet50.in21k_miil_fz': 'RN IN21k-P',
    'hiresnet50.a1_in1k_fz': 'RN A1',
    'hiresnet50.tv2_in1k_fz': 'RN TV2',
    
    # ResNet Semi-SL Frozen models
    'hiresnet50.fb_swsl_ig1b_ft_in1k_fz': 'RN IG1b',
    'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz': 'RN YFCC100m',

    # ResNet SSL Frozen models
    'hiresnet50.in1k_supcon_fz': 'RN SupCon',
    'hiresnet50.in1k_swav_fz': 'RN SwAV',
    'hiresnet50.in1k_mocov3_fz': 'RN MoCo v3',
    'hiresnet50.in1k_spark_fz': 'RN SparK',

    # ViT models
    'hivit_base_patch16_224.orig_in21k': 'ViT',
    'hideit_base_patch16_224.fb_in1k': 'DeiT',
    'hivit_base_patch16_224_miil.in21k': 'ViT IN21k-P',

    'hideit3_small_patch16_224.fb_in1k': 'DeiT 3 Small (IN1k)',
    'hideit3_base_patch16_224.fb_in1k': 'DeiT 3 (IN1k)',
    'hideit3_large_patch16_224.fb_in1k': 'DeiT 3 Large (IN1k)',
    'hideit3_huge_patch16_224.fb_in1k': 'DeiT 3 Huge (IN1k)',

    'hideit3_base_patch16_224.fb_in22k_ft_in1k': 'DeiT 3 (IN21k)',

    'hivit_base_patch16_224.in1k_mocov3': 'ViT MoCo v3',
    'hivit_base_patch16_224.dino': 'ViT DINO',
    'hivit_base_patch16_clip_224.laion2b': 'ViT CLIP',
    'hivit_base_patch16_siglip_224.v2_webli': 'ViT SigLIP v2',
    'hivit_base_patch16_224.mae': 'ViT MAE',

    # ViT Frozen models
    'hivit_base_patch16_224.orig_in21k_fz': 'ViT',
    'hideit_base_patch16_224.fb_in1k_fz': 'DeiT',
    'hivit_base_patch16_224_miil.in21k_fz': 'ViT IN21k-P',
    'hideit3_base_patch16_224.fb_in1k_fz': 'DeiT 3 (IN1k)',
    'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz': 'DeiT 3 (IN21k)',

    'hivit_base_patch16_224.in1k_mocov3_fz': 'ViT MoCo v3',
    'hivit_base_patch16_224.dino_fz': 'ViT DINO',
    'hivit_base_patch16_clip_224.laion2b_fz': 'ViT CLIP',
    'hivit_base_patch16_siglip_224.v2_webli_fz': 'ViT SigLIP v2',
    'hivit_base_patch16_224.mae_fz': 'ViT MAE',
}


VAR_DIC = {
    'setting': 'Setting',
    'acc_mean': 'Accuracy (%)',
    'acc_std': 'Accuracy Std. Dev. (%)',
    'dataset_name': 'Dataset',
    'method': 'Method',
    'model_category': "Architecture",
    'family': 'Method Family',
    'flops': 'Inference FLOPs (10^9)',
    'time_train': 'Train Time (hours)',
    'vram_train': 'Train VRAM (GB)',
    'trainable_percent': 'Task-Specific Parameters (%)',
    'no_params': 'Number of Parameters (10^6)',
    'no_params_trainable': 'Task-Specific Parameters (10^6)',
    'no_params_total': 'Total Parameters (10^6)',
    'no_params_trainable_total': 'Total Task-Specific Params. (10^6)',
    'flops_inference': 'Inference FLOPs (10^9)',
    'tp_stream': 'Stream Throughput (Images/s)',
    'vram_stream': 'Stream VRAM (GB)',
    'latency_stream': 'Stream Latency (s)',
    'tp_batched': 'Batched Throughput  (Images/s)',
    'vram_batched': 'Batched VRAM (GB)',
    'serial' : 'Status',
    'n_cluster_ratio': 'Cluster Ratio'
}


def rename_var(x):
    if x in SETTINGS_DIC.keys():
        return SETTINGS_DIC[x]
    elif x in METHODS_DIC.keys():
        return METHODS_DIC[x]
    elif x in DATASETS_DIC.keys():
        return DATASETS_DIC[x]
    elif x in VAR_DIC.keys():
        return VAR_DIC[x]
    elif x in SERIAL_DIC.keys():
        return SERIAL_DIC[x]
    return x


def rename_vars(df, var_rename=False, args=None):
    if 'setting' in df.columns:
        df['setting'] = df['setting'].apply(rename_var)
    if 'method' in df.columns:
        df['method'] = df['method'].apply(rename_var)
    if 'model_name_extractor' in df.columns:
        df['model_name_extractor'] = df['model_name_extractor'].apply(rename_var)
    if 'dataset_name' in df.columns:
        df['dataset_name'] = df['dataset_name'].apply(rename_var)
    if 'family' in df.columns:
        df['family'] = df['family'].apply(rename_var)
    if 'serial' in df.columns:
        df['serial'] = df['serial'].apply(rename_var)

    if var_rename:
        df.rename(columns=VAR_DIC, inplace=True)
        for k, v in VAR_DIC.items():
            if k == args.x_var_name:
                args.x_var_name = v
            elif k == args.y_var_name:
                args.y_var_name = v
            elif k == args.hue_var_name:
                args.hue_var_name = v
            elif k == args.style_var_name:
                args.style_var_name = v
            elif k == args.size_var_name:
                args.size_var_name = v

    return df


def add_setting(df):
    conditions = [
        (df['serial'] == 21),

        (df['serial'] == 23),
        (df['serial'] == 24),
    ]

    df['setting'] = np.select(conditions, SERIALS_EXPLANATIONS, default='')
    return df


def standarize_df(df):
    # methods
    df = df.fillna({'classifier': '', 'selector': ''})

    df['freeze_backbone_str'] = df['freeze_backbone'].apply(lambda x: '_fz' if x is True else '')

    df['method'] = df['model_name'] + df['freeze_backbone_str']

    df = add_setting(df)
    return df


def keep_columns(df, type='acc'):
    if type == 'all':
        # maybe: 'lr', 'train_loss', 'val_loss'
        kw_list = ['acc', 'cka_', 'l2_', 'dist_', 'MSC', 'intra', 'inter', 'diversity']

        base_cols = [
            'ap_w', 'dataset_name', 'serial', 'setting', 'method',
            'lr', 'n_cluster_ratio', 'batch_size', 'seed', 'model_name_extractor', 'extractor_layer', 'img_per_class',
        ]

        # keep only columns that actually exist
        base_cols = [c for c in base_cols if c in df.columns]

        keep = base_cols + [
            col for col in df.columns
            if any(kw in col for kw in kw_list)
        ]
    
    elif type == 'acc':
        base_cols = [
            'ap_w', 'dataset_name', 'serial', 'setting', 'method',
            'lr', 'n_cluster_ratio', 'extractor_layer', 'img_per_class',
            'model_name_extractor', 'seed',
        ]

        base_cols = [c for c in base_cols if c in df.columns]

        keep = base_cols + [
            col for col in df.columns if 'acc' in col
        ]
    elif type == 'inference_cost':
        keep = ['host', 'serial', 'setting', 'method', 'batch_size', 'throughput',
                'flops', 'max_memory']
    elif type == 'train_cost':
        keep = ['host', 'serial', 'setting', 'method', 'batch_size', 'epochs',
                'dataset_name', 'num_images_train', 'num_images_val',
                'time_total', 'flops', 'max_memory',
                'no_params_trainable', 'no_params']

    df = df[keep]
    return df


def filter_df(df, keep_datasets=None, keep_methods=None, keep_serials=None,
              filter_datasets=None, filter_methods=None, filter_serials=None, keep_ratios=None, keep_extractors=None, keep_ipc=None):
    # print(df['dataset_name'].unique())
    if keep_datasets:
        df = df[df['dataset_name'].isin(keep_datasets)]

    # print(df['dataset_name'].unique())

    if keep_methods:
        df = df[df['method'].isin(keep_methods)]

    if keep_extractors:
        df = df[df['model_name_extractor'].isin(keep_extractors)]

    if keep_serials:
        df = df[df['serial'].isin(keep_serials)]

    # print(df['n_cluster_ratio'].unique())
    if 'n_cluster_ratio' in df.columns:
        df['n_cluster_ratio'] = df['n_cluster_ratio'].fillna(0).astype(int)
    if 'img_per_class' in df.columns:
        df.loc[:, 'img_per_class'] = df['img_per_class'].fillna(1).astype(int)

    if keep_ratios:
        df = df[df['n_cluster_ratio'].isin(keep_ratios)]
        # print(df['n_cluster_ratio'].unique())
    
    if keep_ipc:
        df = df[df['img_per_class'].isin(keep_ipc)]

    if filter_datasets:
        df = df[~df['dataset_name'].isin(filter_datasets)]

    if filter_methods:
        df = df[~df['method'].isin(filter_methods)]

    if filter_serials:
        df = df[~df['serial'].isin(filter_serials)]

    return df


def preprocess_df(
    df, type='acc', keep_datasets=None, keep_methods=None, keep_serials=None,
    filter_datasets=None, filter_methods=None, filter_serials=None, keep_ratios=None, keep_extractors=None, keep_ipc=None):
    # load dataset and preprocess to include method and setting columns, rename val_acc to acc
    df = standarize_df(df)
    # print(df['img_per_class'])
    # drop columns
    df = keep_columns(df, type=type)
    # print(df['img_per_class'])
    # filter
    df = filter_df(df, keep_datasets, keep_methods, keep_serials,
                   filter_datasets, filter_methods, filter_serials, keep_ratios, keep_extractors, keep_ipc)

    df = sort_df(df)
    
    return df


def add_all_cols_group(df, col='dataset_name'):
    subset = df.copy(deep=False)
    subset[col] = 'all'
    df = pd.concat([df, subset], axis=0, ignore_index=True)
    return df


def drop_na(df, args):
    subset = [args.x_var_name, args.y_var_name]
    if args.hue_var_name:
        subset.append(args.hue_var_name)
    if args.style_var_name:
        subset.append(args.style_var_name)
    if args.size_var_name:
        subset.append(args.size_var_name)
    df = df.dropna(subset=subset)
    return df


def sort_df(df, method_only=False, raw_data=False):
    if raw_data:
        df['dataset_order'] = pd.Categorical(df['dataset_name'], categories=DATASETS_DIC.keys(), ordered=True)
        df['model_name_order'] = pd.Categorical(df['model_name'], categories=METHODS_DIC.keys(), ordered=True)
        
        df = df.sort_values(by=['serial', 'dataset_name', 'method_order'], ascending=True)
        df = df.drop(columns=['model_name_order', 'dataset_order'])
    elif method_only:
        df['method_order'] = pd.Categorical(df['method'], categories=METHODS_DIC.keys(), ordered=True)

        df = df.sort_values(by=['serial', 'setting', 'method_order'], ascending=True)
        df = df.drop(columns=['method_order'])
    else:
        df['dataset_order'] = pd.Categorical(df['dataset_name'], categories=DATASETS_DIC.keys(), ordered=True)
        if 'model_name_extractor' in df.columns:
            df['model_name_extractor_order'] = pd.Categorical(df['model_name_extractor'], categories=METHODS_DIC.keys(), ordered=True)
        df['method_order'] = pd.Categorical(df['method'], categories=METHODS_DIC.keys(), ordered=True)
        df['serial_order'] = pd.Categorical(df['serial'], categories=SERIAL_DIC.keys(), ordered=True)
        if 'model_name_extractor_order' in df.columns:
            df = df.sort_values(by=['serial_order', 'dataset_order', 'method_order', 'model_name_extractor_order'], ascending=True)
            df = df.drop(columns=['method_order', 'dataset_order', 'serial_order', 'model_name_extractor_order'])
        else:
            df = df.sort_values(by=['serial_order', 'dataset_order', 'method_order'], ascending=True)
            df = df.drop(columns=['method_order', 'dataset_order', 'serial_order'])
    return df


def sort_cols(df, kw_list=['acc']):
    # Separate matching and non-matching columns
    matching_cols = [col for col in df.columns if any(kw in col for kw in kw_list)]
    non_matching_cols = [col for col in df.columns if not any(kw in col for kw in kw_list)]

    # Reorder the DataFrame
    df = df[matching_cols + non_matching_cols]
    return df

def round_combine_str_mean_std(df, col='acc', dec=4):
    df[f'{col}_mean'] = df[f'{col}_mean'].round(dec)
    df[f'{col}_std'] = df[f'{col}_std'].round(dec)

    df[f'{col}_mean_std_latex'] = df[f'{col}_mean'].astype(str) + "\\pm{" + df[f'{col}_std'].astype(str) + "}"
    df[f'{col}_mean_std'] = df[f'{col}_mean'].astype(str) + "+-" + df[f'{col}_std'].astype(str)

    return df


def group_by_family(x):
    if 'resnet' in x:
        return 'rn'
    elif 'vit' in x or 'deit' in x:
        return 'vit'
    return x
