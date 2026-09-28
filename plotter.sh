#!/bin/bash
# Rebuilds every summary table and figure from the committed CSVs in data/, then the VCIP26 paper and its number check.
# Run from the repo root in the asr env: bash plotter.sh [backbones|pseudo|vcip|few_examples|robustness|paper|verify ...] (default: all)
set -euo pipefail
PY="${PY:-python}"
# a forwarded X display otherwise makes every figure round-trip over the network
export MPLBACKEND=Agg
STAGES=("${@:-backbones pseudo vcip few_examples robustness paper verify}")
STAGES=(${STAGES[@]})
run_stage() { [[ " ${STAGES[*]} " == *" $1 "* ]]; }
plot() { echo "plot $*" | cut -c1-160; $PY -m codebase.plots.plot "$@" > /dev/null; }
has_col() { head -1 "$1" | tr ',' '\n' | grep -qx "$2"; }

all_resnet=('hiresnet50.tv_in1k' 'hiresnet50.tv2_in1k' 'hiresnet50.gluon_in1k' 'hiresnet50.fb_swsl_ig1b_ft_in1k' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k' 'hiresnet50.a1_in1k' 'hiresnet50.in1k_mocov3' 'hiresnet50.in1k_spark' 'hiresnet50.in1k_supcon' 'hiresnet50.in1k_swav' 'hiresnet50.in21k_miil')
all_vit=('hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_224_miil.in21k' 'hideit_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_clip_224.laion2b' 'hivit_base_patch16_224.mae' 'hivit_base_patch16_224.in1k_mocov3' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_siglip_224.v2_webli')
all_models=("${all_resnet[@]}" "${all_vit[@]}")
# frozen-backbone runs carry an _fz suffix on the method name
all_resnet_fz=("${all_resnet[@]/%/_fz}")
all_vit_fz=("${all_vit[@]/%/_fz}")
datasets=('aircraft' 'cub' 'cars')


# ============================================================
# Backbone study: CVGIP25, ACMMMW25, ICME26 (serials 23 FT, 24 FZ)
# ============================================================
if run_stage backbones; then
    $PY -m codebase.analyses.summarize_acc --input_file data/hierarchical_all.csv
    $PY -m codebase.analyses.summarize_acc_metrics --input_file data/hierarchical_all_2.csv
    # CKA, distance and norm metrics (serials 25/27) for the regressions below
    $PY -m codebase.analyses.summarize_acc_metrics --input_file data/hierarchical_all_1.csv --output_file summary_features

    # wAP vs LR, per architecture and per backbone
    for serial in 23 24; do
        [[ $serial == 23 ]] && suffix=ft || suffix=fz
        [[ $serial == 23 ]] && rn=("${all_resnet[@]}") vit=("${all_vit[@]}") || rn=("${all_resnet_fz[@]}") vit=("${all_vit_fz[@]}")
        for arch in resnet vit; do
            [[ $arch == resnet ]] && methods=("${rn[@]}") name=ResNet || methods=("${vit[@]}") name=ViT
            plot --loc_legend 'lower right' --type_plot line --log_scale_x --title "Mean wAP across ${name} ${suffix^^} Backbones vs LR for each Dataset" --fig_size 8 6 --input_file data/hierarchical_stage1.csv --results_dir results_all/cvgip/ --output_file distribution_accross_all_${arch}_${suffix} --keep_methods "${methods[@]}" --keep_serials ${serial} --x_label 'Learning Rate (LR)' --y_label 'Weighted Average Precision(wAP, %)' --keep_datasets "${datasets[@]}" --y_var_name ap_w --x_var_name lr --hue_var_name dataset_name
            for model in "${methods[@]}"; do
                [[ $serial == 23 ]] && out=${model}_ft || out=${model}
                plot --type_plot line --loc_legend 'lower right' --log_scale_x --title "wAP for ${model} ${suffix^^} Backbone vs LR for each Dataset" --fig_size 8 6 --input_file data/hierarchical_stage1.csv --results_dir results_all/cvgip/${arch/resnet/rn}${suffix} --output_file ${out} --keep_methods ${model} --keep_serials ${serial} --x_label 'Learning Rate (LR)' --y_label 'Weighted Average Precision(wAP, %)' --keep_datasets "${datasets[@]}" --y_var_name ap_w --x_var_name lr --hue_var_name dataset_name
            done
        done
    done

    # per-serial box plots of all backbones, top-1 and wAP
    for serial in 23 24; do
        [[ $serial == 23 ]] && rn=("${all_resnet[@]}") vit=("${all_vit[@]}") prefix='Fine Tuned' || rn=("${all_resnet_fz[@]}") vit=("${all_vit_fz[@]}") prefix=Frozen
        for metric in top1 ap_w; do
            [[ $metric == top1 ]] && y=val_acc_level1 ylab='Accuracy(%)' tag=Top1 || y=ap_w ylab='Weighted Average Precision(wAP, %)' tag=wAP
            plot --y_label "$ylab" --font_scale 1.75 --y_var_name $y --x_var_name method --x_rotation 30 --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance --input_file data/hierarchical_all.csv --keep_serials ${serial} --keep_methods "${rn[@]}" --output_file resnet_models_performance_serial_${serial}_${metric} --title "Resnet Models Perfromance ${prefix} (${tag})"
            plot --y_label "$ylab" --font_scale 1.75 --y_var_name $y --x_var_name method --x_rotation 30 --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance --input_file data/hierarchical_all.csv --keep_serials ${serial} --keep_methods "${vit[@]}" --output_file vit_models_performance_serial_${serial}_${metric} --title "ViT Models Perfromance ${prefix} (${tag})"
        done
    done

    # accuracy / ADA ratio vs feature metric regressions, then one grid per accuracy
    metrics=('dist_intra_last_layer_train' 'dist_inter_last_layer_train' 'dist_intra_avg_train' 'dist_inter_avg_train' 'cka_avg_train' 'dist_avg_train' 'dist_norm_avg_train' 'l2_norm_avg_train' 'cka_0_train' 'cka_high_mean_train' 'cka_mid_mean_train' 'cka_low_mean_train' 'dist_intra_0_train' 'dist_inter_0_train')
    reg=(--type_plot reg --fig_size 6.5 4 --summarized --results_dir results_all/new_plots/ap_w --font_scale 3.2 --font_size_correlations 34 --dpi 160 --input_file results_all/acc_metrics/summary_features_ap_w_all.csv --title '')
    # a metric whose column is not in the summary keeps its committed figure
    regplot() { if has_col results_all/acc_metrics/summary_features_ap_w_all.csv "$1"; then plot "${reg[@]}" --x_var_name "$1" --output_file "$2" "${@:3}"; else echo "skip $2: no column $1"; fi; }
    for y_var in acc_max ada_ratio; do
        for metric in "${metrics[@]}"; do
            last=0; [[ $metric == *_last_layer_train ]] && last=1
            for dataset in "${datasets[@]}"; do
                for serial in 23 24; do
                    [[ $serial == 24 ]] && prefix=fz ft_suffix='' || prefix=ft ft_suffix=_ft
                    for arch in rn vit; do
                        [[ $arch == rn ]] && layer=15 models=("${all_resnet[@]}" "${all_resnet_fz[@]}") || layer=11 models=("${all_vit[@]}" "${all_vit_fz[@]}")
                        x=${metric/last_layer/$layer}
                        regplot ${x}${ft_suffix} ${arch}${prefix}_${y_var}_${metric}_${dataset} --y_var_name ${y_var} --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods "${models[@]}"
                    done
                    if [[ $serial == 24 ]]; then
                        models=("${all_resnet_fz[@]}" "${all_vit_fz[@]}")
                        (( last )) && x=cka_last_train || x=${metric}
                    else
                        models=("${all_models[@]}")
                        (( last )) && x=cka_last_train_ft || x=${metric}_ft
                    fi
                    regplot ${x} both${prefix}_${y_var}_${metric}_${dataset} --y_var_name ${y_var} --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods "${models[@]}"
                done
                (( last )) && x=cka_last_train_matched || x=${metric}
                regplot ${x} bothaccftvsmetricfz_${y_var}_${metric}_${dataset} --y_var_name ${y_var} --keep_datasets ${dataset} --keep_serials 23 --keep_methods "${all_models[@]}"
                x=${metric/last_layer_train/last_train}_matched
                regplot ${x} bothft+fz_${y_var}_${metric}_${dataset} --y_var_name ${y_var} --keep_datasets ${dataset} --keep_serials 23 24 --keep_methods "${all_models[@]}" "${all_resnet_fz[@]}" "${all_vit_fz[@]}"
            done
        done
    done
    $PY -m codebase.plots.merge_corr_plots --input_folder results_all/new_plots/ap_w --output_folder results_all/new_plots/merged/wap_multiplot --output_file wap_multiplot --title 'Weighted Accuracy Precision' --x_filter rnfz rnft vitfz vitft bothft bothfz bothaccftvsmetricfz
fi


# ============================================================
# Pseudo-hierarchy on aircraft, CUB, cars: ICME26, CVPRW26
# ============================================================
if run_stage pseudo; then
    # summary tables, then pseudo minus baseline differences
    $PY -m codebase.analyses.summarize_acc --input_file data/hierarchical_epochs.csv --main_serials 64 65 --results_dir results_all/acc/acc_epochs
    $PY -m codebase.analyses.compute_diff --input_csv results_all/acc/acc_epochs/summary_val_acc_level1_main.csv --main_serials 64 65 --output_dir results_all/pseudo_hyperparam/epochs
    $PY -m codebase.analyses.summarize_acc --input_file data/hierarchical_is.csv --main_serials 66 67 --results_dir results_all/acc/acc_is
    $PY -m codebase.analyses.compute_diff --input_csv results_all/acc/acc_is/summary_val_acc_level1_main.csv --main_serials 66 67 --output_dir results_all/pseudo_hyperparam/image_size
    $PY -m codebase.analyses.summarize_acc --input_file data/hierarchical_augs.csv --main_serials 68 69 --results_dir results_all/acc/acc_augs
    $PY -m codebase.analyses.compute_diff --input_csv results_all/acc/acc_augs/summary_val_acc_level1_main.csv --main_serials 68 69 --output_dir results_all/pseudo_hyperparam/augs
    $PY -m codebase.analyses.summarize_acc --input_file data/hierarchical_model_size.csv --main_serials 70 --results_dir results_all/acc/acc_model_size
    $PY -m codebase.analyses.compute_diff --input_csv results_all/acc/acc_model_size/summary_val_acc_level1_main.csv --main_serials 70 --output_dir results_all/pseudo_hyperparam/model_size
    $PY -m codebase.analyses.summarize_acc --input_file data/hierarchical_algo.csv --main_serials 40 52 53 --results_dir results_all/acc/acc_algo
    $PY -m codebase.analyses.compute_diff --input_csv results_all/acc/acc_algo/summary_val_acc_level1_main.csv --main_serials 40 52 53 --output_dir results_all/pseudo_hyperparam/clus_algo
    $PY -m codebase.analyses.summarize_acc --input_file data/hierarchical_diff_extractor.csv --main_serials 40 51 --results_dir results_all/acc/acc_extractor
    $PY -m codebase.analyses.compute_diff --input_csv results_all/acc/acc_extractor/summary_val_acc_level1_main.csv --main_serials 40 51 --output_dir results_all/pseudo_hyperparam/diff_extractor
    $PY -m codebase.analyses.summarize_acc --input_file data/hierarchical_pseudo_2.csv --main_serials 40 23 32 --results_dir results_all/acc/acc_pseudo_main
    $PY -m codebase.analyses.compute_diff --input_csv results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --main_serials 32 23 40 --output_dir results_all/pseudo_hyperparam/main_results

    # hyperparameter sweeps
    plot --keep_serials 68 69 --x_var_name serial --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_augs.csv --output_file compare_augs --title 'Comparison of 12 augmentations of Aircraft with hideit3_base_patch16_224.fb_in1k model in Baseline vs Pseudo-Hierarchy settings'
    plot --keep_serials 66 67 --x_var_name serial --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_is.csv --output_file compare_is --title 'Comparison of 10 image sizes (is) of Aircraft with hideit3_base_patch16_224.fb_in1k model in Baseline vs Pseudo-Hierarchy settings'
    plot --keep_serials 64 65 --x_var_name serial --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_epochs.csv --output_file compare_epochs --title 'Comparison of 10 epochs of Aircraft with hideit3_base_patch16_224.fb_in1k model in Baseline vs Pseudo-Hierarchy settings'
    plot --keep_serials 70 --x_var_name model_name_extractor --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_model_size.csv --output_file compare_model_size --title 'Comparison of hideit3_base_patch16_224.fb_in1k model sizes on Aircraft in Baseline vs Pseudo-Hierarchy settings'
    plot --keep_serials 70 --summarized --x_var_name model_name_extractor --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --type_plot bar --input_file results_all/acc/acc_model_size/summary_val_acc_level1_main.csv --output_file compare_model_size_ada --title 'Comparison of hideit3_base_patch16_224.fb_in1k model sizes on Aircraft in Baseline vs Pseudo-Hierarchy settings'
    plot --keep_serials 70 --font_scale 1.5 --fig_size 10 6 --summarized --keep_extractor 'hideit3_base_patch16_224.fb_in1k' 'hideit3_large_patch16_224.fb_in1k' 'hideit3_huge_patch16_224.fb_in1k' --x_var_name model_name_extractor --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --type_plot bar --input_file results_all/acc/acc_model_size/summary_val_acc_level1_main.csv --output_file compare_model_size_ada_alt --title 'hideit3_base_patch16_224.fb_in1k sizes on Aircraft in Pseudo-Hierarchy'

    # clustering algorithm and feature extractor
    plot --keep_serials 40 52 --x_ticks_labels 'HPH' 'Kmean' --font_scale 1.75 --fig_size 8 6 --keep_ratios 50 --keep_dataset aircraft_pl --keep_methods hideit3_base_patch16_224.fb_in1k --x_var_name serial --y_label 'Accuracy (%)' --y_var_name val_acc_level1 --type_plot box --input_file data/hierarchical_algo.csv --output_file compare_algo_all --title 'Clustering algorithms on Aircraft with hideit3_base_patch16_224.fb_in1k'
    plot --keep_serials 40 52 --keep_ratios 50 --keep_dataset aircraft_pl --keep_methods hideit3_base_patch16_224.fb_in1k --x_var_name serial --y_label 'Accuracy (%)' --y_var_name val_acc_level1 --type_plot box --input_file data/hierarchical_algo.csv --output_file compare_algo_only_kmeans --title 'Comparison of different clustering algorithm on Aircraft with hideit3_base_patch16_224.fb_in1k model'
    for arch in rn vit; do
        [[ $arch == rn ]] && ext=("${all_resnet[@]}") name=Resnet || ext=("${all_vit[@]}") name=ViT
        plot --keep_serials 40 51 --keep_ratios 50 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --keep_dataset aircraft_pl --keep_methods hideit3_base_patch16_224.fb_in1k --keep_extractor "${ext[@]}" --x_var_name model_name_extractor --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_diff_extractor.csv --output_file compare_extractors_${arch} --title "Overall performance of ${name} extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model"
        plot --keep_serials 40 51 --keep_ratios 50 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --summarized --keep_methods hideit3_base_patch16_224.fb_in1k --keep_dataset aircraft_pl --y_var_name acc_std --y_label 'Accuracy Standard Deviation' --keep_extractor "${ext[@]}" --x_var_name model_name_extractor --type_plot bar --input_file results_all/acc/acc_extractor/summary_val_acc_level1_main.csv --output_file compare_extractors_std_${arch} --title "StD performance of ${name} extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model"
        plot --keep_serials 40 51 --keep_ratios 50 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --summarized --keep_methods hideit3_base_patch16_224.fb_in1k --keep_dataset aircraft_pl --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --keep_extractor "${ext[@]}" --x_var_name model_name_extractor --type_plot bar --input_file results_all/acc/acc_extractor/summary_val_acc_level1_main.csv --output_file compare_extractors_ada_${arch} --title "ADA ratio of ${name} extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model"
    done
    plot --keep_serials 40 51 --keep_ratios 50 --fig_size 24 6 --x_rotation 20 --font_scale 1.75 --summarized --keep_methods hideit3_base_patch16_224.fb_in1k --keep_dataset aircraft_pl --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --keep_extractor "${all_models[@]}" --x_var_name model_name_extractor --type_plot bar --input_file results_all/acc/acc_extractor/summary_val_acc_level1_main.csv --output_file compare_extractors_ada_all --title 'CV of all extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model'

    # baseline (32) vs real hierarchy (23) vs pseudo-hierarchy (40)
    main_acc=(--input_file data/hierarchical_pseudo_2.csv --y_var_name val_acc_level1 --y_label 'Accuracy (%)')
    main_ada=(--input_file results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --summarized --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)')
    plot --keep_serials 32 23 40 --loc_legend 'lower right' --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name serial "${main_acc[@]}" --hue_var_name dataset_name --type_plot box --output_file compare_main_plot_acc --title 'Overall performance of different models on 3 settings: Baseline, Real Hierarchy, Pseudo-Hierarchy'
    plot --keep_serials 32 23 40 --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name serial "${main_ada[@]}" --hue_var_name dataset_name --type_plot box --output_file compare_main_plot_ada --title 'CV of different models on 3 settings: Baseline, Real Hierarchy, Pseudo-Hierarchy'
    plot --keep_serials 40 --keep_ratios 25 50 70 --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name n_cluster_ratio "${main_ada[@]}" --hue_var_name dataset_name --type_plot box --output_file compare_main_plot_ratios_ada --title 'CV of different pseudo-hierarchy ratios on 3 datasets'
    plot --keep_serials 40 --loc_legend 'lower right' --keep_ratios 25 50 70 --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name n_cluster_ratio "${main_acc[@]}" --hue_var_name dataset_name --type_plot box --output_file compare_main_plot_ratios_acc --title 'Overall performance of different pseudo-hierarchy ratios on 3 datasets'
    plot --keep_serials 32 23 40 --keep_ratios 0 50 --loc_legend 'lower right' --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name dataset_name "${main_acc[@]}" --hue_var_name serial --type_plot box --output_file compare_main_plot_acc_alt --title 'Overall performance of different models on 3 settings: Baseline, Real Hierarchy, Pseudo-Hierarchy'
    plot --keep_serials 32 23 40 --keep_ratios 0 50 --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name dataset_name "${main_ada[@]}" --hue_var_name serial --type_plot box --output_file compare_main_plot_ada_alt --title 'CV of different models on 3 settings: Baseline, Real Hierarchy, Pseudo-Hierarchy'
    plot --keep_serials 40 --keep_ratios 25 50 70 --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name dataset_name "${main_ada[@]}" --hue_var_name n_cluster_ratio --palette pastel --type_plot box --output_file compare_main_plot_ratios_ada_alt --title 'CV of different pseudo-hierarchy ratios on 3 datasets'
    plot --keep_serials 40 --loc_legend 'lower right' --keep_ratios 25 50 70 --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name dataset_name "${main_acc[@]}" --hue_var_name n_cluster_ratio --palette pastel --type_plot box --output_file compare_main_plot_ratios_acc_alt --title 'Overall performance of different pseudo-hierarchy ratios on 3 datasets'
    for dataset in aircraft cars cub; do
        for ratio in 25 50 70; do
            plot --keep_serials 32 23 40 --font_scale 1.75 --fig_size 13 8 --x_rotation 20 --keep_ratios 0 ${ratio} --loc_legend 'lower left' --keep_methods "${all_resnet[@]}" --keep_dataset ${dataset} --aggregate_dataset --x_var_name method "${main_acc[@]}" --hue_var_name serial --type_plot box --output_file compare_main_plot_acc_${dataset}_${ratio}_rn --title "Accuracy vs ResNet Backbones for ${dataset}"
            plot --keep_serials 32 23 40 --font_scale 1.75 --fig_size 13 8 --x_rotation 20 --keep_ratios 0 ${ratio} --loc_legend 'lower left' --keep_methods "${all_vit[@]}" --keep_dataset ${dataset} --aggregate_dataset --x_var_name method "${main_acc[@]}" --hue_var_name serial --type_plot box --output_file compare_main_plot_acc_${dataset}_${ratio}_vit --title "Accuracy vs ViT Backbones for ${dataset}"
        done
        plot --keep_serials 40 --font_scale 1.75 --fig_size 8 6 --palette pastel --keep_ratios 25 50 70 --keep_dataset ${dataset} --aggregate_dataset --x_var_name dataset_name "${main_ada[@]}" --hue_var_name n_cluster_ratio --type_plot box --output_file compare_main_plot_ratios_ada_${dataset} --title "${dataset} Cluster Ratio Comparison"
        plot --keep_serials 40 --font_scale 1.75 --fig_size 8 6 --palette pastel --loc_legend 'lower right' --keep_ratios 25 50 70 --keep_dataset ${dataset} --aggregate_dataset --x_var_name dataset_name "${main_acc[@]}" --hue_var_name n_cluster_ratio --type_plot box --output_file compare_main_plot_ratios_acc_${dataset} --title "${dataset} Cluster Ratio Comparison"
    done
    for ratio in 25 50 70; do
        for arch in rn vit; do
            [[ $arch == rn ]] && ext=("${all_resnet[@]}") || ext=("${all_vit[@]}")
            plot --keep_serials 40 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --loc_legend 'lower right' --keep_ratios ${ratio} --keep_extractor "${ext[@]}" --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name method "${main_acc[@]}" --hue_var_name dataset_name --type_plot box --output_file compare_main_plot_${ratio}_acc_${arch} --title "Overall performance of different pseudo-hierarchy with ${ratio}% ratios on 3 datasets"
            plot --keep_serials 40 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --keep_ratios ${ratio} --keep_dataset aircraft cub cars --keep_extractor "${ext[@]}" --aggregate_dataset --x_var_name method "${main_ada[@]}" --hue_var_name dataset_name --type_plot bar --output_file compare_main_plot_${ratio}_ada_${arch} --title "CV of different pseudo-hierarchy with ${ratio}% ratios on 3 datasets"
        done
    done

    # SoyLocal as first run (serial 39 pseudo, 32 baseline); VCIP26 reuses compare_soylocal.png
    plot --keep_serials 39 32 --font_scale 1.75 --fig_size 8 6 --loc_legend 'lower right' --keep_dataset soylocal --x_var_name n_cluster_ratio --y_var_name val_acc_level1 --x_label 'Cluster Ratio (%)' --y_label 'Accuracy (%)' --hue_var_name serial --type_plot box --input_file data/hierarchical_pseudo.csv --output_file compare_soylocal --title 'Accuracy of backbones on Soylocal dataset'
fi


# ============================================================
# SoyLocal and other leaf datasets: VCIP26
# ============================================================
if run_stage vcip; then
    $PY -m codebase.analyses.summarize_acc --input_file data/hierarchical_soylocal.csv --main_serials 55 54 --results_dir results_all/acc/acc_soylocal
    $PY -m codebase.analyses.compute_diff --input_csv results_all/acc/acc_soylocal/summary_val_acc_level1_main.csv --main_serials 55 54 --output_dir results_all/pseudo_hyperparam/soylocal
    $PY -m codebase.analyses.summarize_acc --input_file data/other_soy_stuff.csv --main_serials 83 86 --results_dir results_all/acc/acc_othersoy
    $PY -m codebase.analyses.compute_diff --input_csv results_all/acc/acc_othersoy/summary_val_acc_level1_main.csv --main_serials 83 86 --output_dir results_all/pseudo_hyperparam/othersoy

    plot --keep_serials 55 54 --fig_size 10 6 --loc_legend 'lower right' --font_scale 1.2 --keep_dataset soylocal --x_rotation 20 --keep_methods "${all_resnet[@]}" --x_var_name method --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name serial --type_plot box --input_file data/hierarchical_soylocal.csv --output_file compare_resnet_soylocal --title 'Accuracy of Resnet backbones on Soylocal dataset'
    plot --keep_serials 55 54 --fig_size 10 6 --loc_legend 'lower right' --font_scale 1.2 --keep_dataset soylocal --x_rotation 20 --keep_methods "${all_vit[@]}" --x_var_name method --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name serial --type_plot box --input_file data/hierarchical_soylocal.csv --output_file compare_vit_soylocal --title 'Accuracy of ViT backbones on Soylocal dataset'
    plot --keep_serials 55 54 --font_scale 1.5 --fig_size 8 6 --keep_dataset soylocal --x_var_name n_cluster_ratio --y_var_name lr --x_label 'Cluster Ratio (%)' --y_label 'Learning Rates' --hue_var_name val_acc_level1 --input_file data/hierarchical_soylocal.csv --output_file heatmap_soylocal --title 'Heatmap of Soylocal dataset' --type_plot heatmap

    for model in hivit_base_patch16_224.orig_in21k hideit_base_patch16_224.fb_in1k; do
        [[ $model == hivit* ]] && tag=vit name=ViT || tag=deit name=Deit
        plot --keep_serials 84 85 --keep_method ${model} --font_scale 1.5 --fig_size 8 6 --keep_dataset cotton --x_var_name n_cluster_ratio --y_var_name seed --y_label 'Seeds' --x_label 'Cluster Ratio (%)' --hue_var_name val_acc_level1 --input_file data/cotton_seeds.csv --output_file heatmap_cotton_seed_${tag} --title "Heatmap of Cotton with different seeds on ${name}" --type_plot heatmap
        for dataset in cotton soyageing_r1 soygene soyglobal soyageing; do
            plot --keep_serials 83 86 --keep_method ${model} --font_scale 1.5 --fig_size 8 6 --keep_dataset ${dataset} --x_var_name n_cluster_ratio --y_var_name val_acc_level1 --x_label 'Cluster Ratio (%)' --input_file data/other_soy_stuff.csv --results_dir results_all/plots/other_leaves --output_file box_${dataset}_${tag} --title "${dataset} on ${name}" --type_plot box
        done
        # same-class mix and hierarchical label smoothing (serials 90-95)
        plot --keep_serials 54 55 90 91 92 93 94 95 --keep_method ${model} --font_scale 1.5 --fig_size 10 6 --x_rotation 25 --keep_dataset soylocal --x_var_name serial --y_var_name val_acc_level1 --input_file data/label_smooth_cutmix.csv --results_dir results_all/plots/label_smooth_cutmix --output_file box_soylocal_${tag} --title "Soylocal on ${name}" --type_plot box
        plot --keep_serials 83 86 90 91 92 93 94 95 --keep_method ${model} --font_scale 1.5 --fig_size 10 6 --x_rotation 25 --keep_dataset cotton --x_var_name serial --y_var_name val_acc_level1 --input_file data/label_smooth_cutmix.csv --results_dir results_all/plots/label_smooth_cutmix --output_file box_cotton_${tag} --title "Cotton on ${name}" --type_plot box
    done
fi


# ============================================================
# Few-example study (serials 102 baseline, 103 real hierarchy)
# ============================================================
if run_stage few_examples; then
    $PY -m codebase.analyses.summarize_acc --input_file data/model_low_examples.csv --main_serials 102 103 --results_dir results_all/acc/acc_few_examples
    # dataset / images-per-class pairs that have runs (missing img_per_class means 1)
    pairs=$($PY -c "import pandas as pd; d = pd.read_csv('data/model_low_examples.csv'); d = d[~d.dataset_name.str.endswith('_baseline')]; print(' '.join(sorted({f'{a}:{int(b)}' for a, b in zip(d.dataset_name, d.img_per_class.fillna(1))})))")
    for pair in ${pairs}; do
        dataset=${pair%:*} ipc=${pair#*:}
        for arch in resnet vit; do
            [[ $arch == resnet ]] && models=("${all_resnet[@]}") name=Resnet || models=("${all_vit[@]}") name=ViT
            plot --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc ${ipc} --keep_method "${models[@]}" --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name val_acc_level1 --input_file data/model_low_examples.csv --results_dir results_all/plots/model_low_examples --output_file box_${ipc}_example_${dataset}_${arch} --title "${ipc}-Example ${name} Models on ${dataset}" --type_plot box
            plot --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc ${ipc} --keep_method "${models[@]}" --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name ada_ratio --input_file results_all/acc/acc_few_examples/summary_val_acc_level1_main.csv --results_dir results_all/plots/model_low_examples --output_file bar_${ipc}_example_${dataset}_${arch} --title "${ipc}-Example ${name} Models on ${dataset}" --type_plot bar --summarized
        done
    done
fi


# ============================================================
# Robustness: hierarchical vs baseline per optimizer, ranking stability, seed noise
# ============================================================
if run_stage robustness; then
    $PY -m codebase.analyses.robustness
fi


# ============================================================
# VCIP26 paper: build from the figures as submitted, then check its numbers and figures against results_all/
# ============================================================
if run_stage paper; then
    bash drafts/VCIP26/build.sh
fi
if run_stage verify; then
    $PY -m codebase.verify.verify_paper_numbers
fi


# ============================================================
# wandb downloads (need an API key with access to nycu_pcs/Hierarchical); the CSVs above are their committed output
# ============================================================
# $PY -m codebase.download.dump_wandb_runs
# $PY -m codebase.verify.reconcile_serials
# $PY -m codebase.download.download_save_wandb_data --serials 23 24 --output_file hierarchical_stage1.csv
# $PY -m codebase.download.download_save_wandb_data_feature_metrics --serials 26 28 --output_file hierarchical_feature_metrics_2.csv
# $PY -m codebase.download.merge_acc_metrics_stats --input_file_metrics data/hierarchical_feature_metrics_2.csv --output_file hierarchical_all_2.csv
# $PY -m codebase.download.download_save_wandb_data --serials 64 65 --output_file hierarchical_epochs.csv
# $PY -m codebase.download.download_save_wandb_data --serials 66 67 --output_file hierarchical_is.csv
# $PY -m codebase.download.download_save_wandb_data --serials 68 69 --output_file hierarchical_augs.csv
# $PY -m codebase.download.download_save_wandb_data --serials 70 --output_file hierarchical_model_size.csv
# $PY -m codebase.download.download_save_wandb_data --serials 40 52 53 --output_file hierarchical_algo.csv
# $PY -m codebase.download.download_save_wandb_data --serials 40 51 --output_file hierarchical_diff_extractor.csv
# $PY -m codebase.download.download_save_wandb_data --serials 40 23 32 --output_file hierarchical_pseudo_2.csv
# $PY -m codebase.download.download_save_wandb_data --serials 54 55 --output_file hierarchical_soylocal.csv
# $PY -m codebase.download.download_save_wandb_data --serials 83 86 --output_file other_soy_stuff.csv
# $PY -m codebase.download.download_save_wandb_data --serials 83 86 54 55 90 91 92 93 94 95 --output_file label_smooth_cutmix.csv
# $PY -m codebase.download.download_save_wandb_data --serials 102 103 --output_file model_low_examples.csv
