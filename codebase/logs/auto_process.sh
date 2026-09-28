#!/bin/bash

# ============================================================
# Data Download & Preprocessing
# ============================================================

# Download feature metrics for serials 26 and 28
# python download_save_wandb_data_feature_metrics.py --serials 26 28 --output_file hierarchical_feature_metrics_2.csv

# Merge accuracy, correlation metrics, and pretraining stats
# python merge_acc_metrics_stats.py --input_file_metrics data/hierarchical_feature_metrics_2.csv --output_file hierarchical_all_2.csv

# download base serials 23 and 24
# python download_save_wandb_data.py --serials 23 24 --output_file hierarchical_stage1.csv

# LR script placeholder
# python lr_script.py


# ============================================================
# Accuracy & Metric Summarization
# ============================================================

# summarize standard accuracy
# python summarize_acc.py --input_file data/hierarchical_all_2.csv

# Summarize accuracy metrics (correlations, CKA, etc.)
python summarize_acc_metrics.py --input_file data/hierarchical_all_2.csv

# summarize compute cost (FLOPs, param count)
# python summarize_cost.py
# python summarize_cost.py --acc_to_use val_acc_level2 --output_file cost_val_acc_level2
# python summarize_cost.py --acc_to_use ap_w --output_file cost_ap_w


# ============================================================
# Model Arrays
# ============================================================

# Target datasets and training serials
datasets_array=('aircraft' 'cub' 'cars')
serials=('23' '24')

# ResNet backbones (fine-tuned + frozen)
all_resnet=('hiresnet50.tv_in1k' 'hiresnet50.tv2_in1k' 'hiresnet50.gluon_in1k' 'hiresnet50.fb_swsl_ig1b_ft_in1k' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k' 'hiresnet50.a1_in1k' 'hiresnet50.in1k_mocov3' 'hiresnet50.in1k_spark' 'hiresnet50.in1k_supcon' 'hiresnet50.in1k_swav' 'hiresnet50.in21k_miil' 'hiresnet50.tv_in1k_fz' 'hiresnet50.tv2_in1k_fz' 'hiresnet50.gluon_in1k_fz' 'hiresnet50.fb_swsl_ig1b_ft_in1k_fz' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz' 'hiresnet50.a1_in1k_fz' 'hiresnet50.in1k_mocov3_fz' 'hiresnet50.in1k_spark_fz' 'hiresnet50.in1k_supcon_fz' 'hiresnet50.in1k_swav_fz' 'hiresnet50.in21k_miil_fz')

# ViT backbones (fine-tuned + frozen)
all_vit=('hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_224_miil.in21k' 'hideit_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_clip_224.laion2b' 'hivit_base_patch16_224.mae' 'hivit_base_patch16_224.in1k_mocov3' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_siglip_224.v2_webli' 'hivit_base_patch16_224.orig_in21k_fz' 'hivit_base_patch16_224_miil.in21k_fz' 'hideit_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz' 'hivit_base_patch16_clip_224.laion2b_fz' 'hivit_base_patch16_224.mae_fz' 'hivit_base_patch16_224.in1k_mocov3_fz' 'hivit_base_patch16_224.dino_fz' 'hivit_base_patch16_siglip_224.v2_webli_fz')

# Fine-tuned only subsets
all_resnet_nofz=('hiresnet50.tv_in1k' 'hiresnet50.tv2_in1k' 'hiresnet50.gluon_in1k' 'hiresnet50.fb_swsl_ig1b_ft_in1k' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k' 'hiresnet50.a1_in1k' 'hiresnet50.in1k_mocov3' 'hiresnet50.in1k_spark' 'hiresnet50.in1k_supcon' 'hiresnet50.in1k_swav' 'hiresnet50.in21k_miil')
all_vit_nofz=('hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_224_miil.in21k' 'hideit_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_clip_224.laion2b' 'hivit_base_patch16_224.mae' 'hivit_base_patch16_224.in1k_mocov3' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_siglip_224.v2_webli')

# Frozen only subsets
all_resnet_onlyfz=('hiresnet50.tv_in1k_fz' 'hiresnet50.tv2_in1k_fz' 'hiresnet50.gluon_in1k_fz' 'hiresnet50.fb_swsl_ig1b_ft_in1k_fz' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz' 'hiresnet50.a1_in1k_fz' 'hiresnet50.in1k_mocov3_fz' 'hiresnet50.in1k_spark_fz' 'hiresnet50.in1k_supcon_fz' 'hiresnet50.in1k_swav_fz' 'hiresnet50.in21k_miil_fz')
all_vit_onlyfz=('hivit_base_patch16_224.orig_in21k_fz' 'hivit_base_patch16_224_miil.in21k_fz' 'hideit_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz' 'hivit_base_patch16_clip_224.laion2b_fz' 'hivit_base_patch16_224.mae_fz' 'hivit_base_patch16_224.in1k_mocov3_fz' 'hivit_base_patch16_224.dino_fz' 'hivit_base_patch16_siglip_224.v2_webli_fz')

# Combined model arrays
all_models=("${all_resnet[@]}" "${all_vit[@]}")
all_models_ft=("${all_resnet_nofz[@]}" "${all_vit_nofz[@]}")
all_models_fz=("${all_resnet_onlyfz[@]}" "${all_vit_onlyfz[@]}")

# ResNet grouped by pretraining type
resnet_fsl=('hiresnet50.tv_in1k' 'hiresnet50.tv2_in1k' 'hiresnet50.gluon_in1k' 'hiresnet50.a1_in1k' 'hiresnet50.in21k_miil' 'hiresnet50.tv_in1k_fz' 'hiresnet50.tv2_in1k_fz' 'hiresnet50.gluon_in1k_fz' 'hiresnet50.a1_in1k_fz' 'hiresnet50.in21k_miil_fz')
resnet_semisl=('hiresnet50.fb_swsl_ig1b_ft_in1k' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k' 'hiresnet50.fb_swsl_ig1b_ft_in1k_fz' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz')
resnet_generative=('hiresnet50.in1k_spark' 'hiresnet50.in1k_spark_fz')
resnet_discrim=('hiresnet50.in1k_mocov3' 'hiresnet50.in1k_supcon' 'hiresnet50.in1k_swav' 'hiresnet50.in1k_mocov3_fz' 'hiresnet50.in1k_supcon_fz' 'hiresnet50.in1k_swav_fz')

# ViT grouped by pretraining type
vit_fsl=('hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_224_miil.in21k' 'hideit_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_224.orig_in21k_fz' 'hivit_base_patch16_224_miil.in21k_fz' 'hideit_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz')
vit_generative=('hivit_base_patch16_224.mae' 'hivit_base_patch16_224.mae_fz')
vit_discrim=('hivit_base_patch16_clip_224.laion2b' 'hivit_base_patch16_224.in1k_mocov3' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_siglip_224.v2_webli' 'hivit_base_patch16_clip_224.laion2b_fz' 'hivit_base_patch16_224.in1k_mocov3_fz' 'hivit_base_patch16_224.dino_fz' 'hivit_base_patch16_siglip_224.v2_webli_fz')


# ============================================================
# -- All Models Performance Plots --
# ============================================================

# per-serial box plots for all ResNet and ViT models (Top-1 and wAP)
base_cmd_top1="python plot.py --y_label 'Accuracy(%)' --font_scale 1.75 --y_var_name val_acc_level1 --x_var_name method --x_rotation 30 --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance"
base_cmd_apw="python plot.py --y_label 'Weighted Average Precision(wAP, %)' --font_scale 1.75 --y_var_name ap_w --x_var_name method --x_rotation 30 --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance"
for serial in "${serials[@]}"; do
    if [[ "$serial" -eq 24 ]]; then
        resnet_array=${all_resnet_onlyfz[*]}; vit_array=${all_vit_onlyfz[*]}; prefix="Frozen"
    else
        resnet_array=${all_resnet_nofz[*]}; vit_array=${all_vit_nofz[*]}; prefix="Fine Tuned"
    fi
    # ResNet top-1
    cmd="${base_cmd_top1} --input_file data/hierarchical_all.csv --keep_serials ${serial} --keep_methods ${resnet_array[*]} --output_file resnet_models_performance_serial_${serial}_top1 --title 'Resnet Models Perfromance ${prefix} (Top1)'"
    echo "Running: ${cmd}"; eval "${cmd}"
    # ViT top-1
    cmd="${base_cmd_top1} --input_file data/hierarchical_all.csv --keep_serials ${serial} --keep_methods ${vit_array[*]} --output_file vit_models_performance_serial_${serial}_top1 --title 'ViT Models Perfromance ${prefix} (Top1)'"
    echo "Running: ${cmd}"; eval "${cmd}"
    # ResNet wAP
    cmd="${base_cmd_apw} --input_file data/hierarchical_all.csv --keep_serials ${serial} --keep_methods ${resnet_array[*]} --output_file resnet_models_performance_serial_${serial}_ap_w --title 'Resnet Models Perfromance ${prefix} (wAP)'"
    echo "Running: ${cmd}"; eval "${cmd}"
    # ViT wAP
    cmd="${base_cmd_apw} --input_file data/hierarchical_all.csv --keep_serials ${serial} --keep_methods ${vit_array[*]} --output_file vit_models_performance_serial_${serial}_ap_w --title 'ViT Models Perfromance ${prefix} (wAP)'"
    echo "Running: ${cmd}"; eval "${cmd}"
done


# ============================================================
# Metric & Y-Axis Configuration
# ============================================================

# Y-axis options and accuracy metric to use
y_ax=('acc_max' 'ada_ratio')
accuracies=('ap_w')
methods=('resnet' 'vit')

# Active metrics list (intra/inter distances + CKA variants)
metrics=('dist_intra_last_layer_train' 'dist_inter_last_layer_train'
         'dist_intra_avg_train' 'dist_inter_avg_train' 'cka_avg_train' 'dist_avg_train' 'dist_norm_avg_train'
         'l2_norm_avg_train' 'cka_0_train' 'cka_high_mean_train'
         'cka_mid_mean_train' 'cka_low_mean_train' 'dist_intra_0_train' 'dist_inter_0_train')

# full metric list including CKA last layer
# metrics=('acc_in1k' 'MSC_train' 'MSC_test' 'V_intra_train' 'V_intra_test' 'S_inter_train' 'S_inter_test'
#          'cis_clustering_diversity_train' 'cis_clustering_diversity_test' 'cis_spectral_diversity_train' 'cis_spectral_diversity_test'
#          'cka_avg_train' 'cka_avg_test' 'dist_avg_train' 'dist_avg_test' 'dist_norm_avg_train'
#          'dist_norm_avg_test' 'l2_norm_avg_train' 'l2_norm_avg_test'
#          'cka_0_train' 'cka_0_test' 'cka_high_mean_train' 'cka_mid_mean_train' 'cka_low_mean_train'
#          'cka_high_mean_test' 'cka_mid_mean_test' 'cka_low_mean_test' 'clustering_diversity_train' 'spectral_diversity_train'
#          'cka_last_layer_train' 'cka_last_layer_test'
#          'cis_cka_0_train' 'cis_cka_0_test' 'cis_cka_last_train' 'cis_cka_last_test'
#          'cis_dist_0_train' 'cis_dist_0_test' 'cis_dist_last_train' 'cis_dist_last_test')

# train-only metric list
# metrics=('acc_in1k' 'MSC_train' 'V_intra_train' 'S_inter_train'
#          'cis_clustering_diversity_train' 'cis_spectral_diversity_train'
#          'cka_avg_train' 'dist_avg_train' 'dist_norm_avg_train'
#          'l2_norm_avg_train' 'cka_0_train' 'cka_high_mean_train'
#          'cka_mid_mean_train' 'cka_low_mean_train' 'clustering_diversity_train'
#          'spectral_diversity_train' 'cka_last_layer_train'
#          'cis_cka_0_train' 'cis_cka_last_train' 'cis_dist_0_train' 'cis_dist_last_train')

# newer CIS-CKA variant metrics
# metrics=(
#         'cis_cka_high_mean_train'
#         'cis_cka_low_mean_train'
#         'cka_inv_low_mean_train' 'cka_inv_high_mean_train'
#         'cis_cka_inv_low_mean_train' 'cis_cka_inv_high_mean_train'
#         )


# ============================================================
# -- Correlation Regression Plots --
# ============================================================

# full nested loop for regression plots across
#            all accuracy metrics, y-axes, metrics, datasets, serials, and architectures
for accuracy in "${accuracies[@]}"; do
    input_path="results_all/acc_metrics/summary_${accuracy}_all.csv"

    for y_var in "${y_ax[@]}"; do
        for metric in "${metrics[@]}"; do
            # Determine CKA last-layer state and base command
            if [[ "$metric" == "cka_last_layer_train" || "$metric" == "dist_intra_last_layer_train" || "$metric" == "dist_inter_last_layer_train" ]]; then
                state="train"
                base_cmd="python plot.py --type_plot reg --fig_size 6.5 4 --summarized --results_dir results_all/new_plots/${accuracy} --font_scale 3.2 --font_size_correlations 34 --dpi 160"
            elif [[ "$metric" == "cka_last_layer_test" || "$metric" == "dist_intra_last_layer_test" || "$metric" == "dist_inter_last_layer_test" ]]; then
                state="test"
                base_cmd="python plot.py --type_plot reg --fig_size 6.5 4 --summarized --results_dir results_all/new_plots/${accuracy} --font_scale 3.2 --font_size_correlations 34 --dpi 160"
            else
                state=""
                base_cmd="python plot.py --type_plot reg --fig_size 6.5 4 --summarized --results_dir results_all/new_plots/${accuracy} --font_scale 3.2 --font_size_correlations 34 --dpi 160"
            fi

            # -- Per-architecture, per-dataset, per-serial plots --
            for dataset in "${datasets_array[@]}"; do
                for serial in "${serials[@]}"; do
                    for method in "${methods[@]}"; do
                        # Select backbone list and resolve last-layer x_var per architecture
                        if [[ "$method" == "resnet" ]]; then
                            model_used=${all_resnet[*]}
                            model_label="rn"
                            # ResNet has 16 layers (index 15)
                            if [[ "$metric" == "cka_last_layer_train" ]]; then x_var="cka_15_train"
                            elif [[ "$metric" == "dist_intra_last_layer_train" ]]; then x_var="dist_intra_15_train"
                            elif [[ "$metric" == "dist_inter_last_layer_train" ]]; then x_var="dist_inter_15_train"
                            elif [[ "$metric" == "cka_last_layer_test" ]]; then x_var="cka_15_test"
                            else x_var="${metric}"; fi
                        else
                            model_used=${all_vit[*]}
                            model_label="vit"
                            # ViT has 12 layers (index 11)
                            if [[ "$metric" == "cka_last_layer_train" ]]; then x_var="cka_11_train"
                            elif [[ "$metric" == "dist_intra_last_layer_train" ]]; then x_var="dist_intra_11_train"
                            elif [[ "$metric" == "dist_inter_last_layer_train" ]]; then x_var="dist_inter_11_train"
                            elif [[ "$metric" == "cka_last_layer_test" ]]; then x_var="cka_11_test"
                            else x_var="${metric}"; fi
                        fi

                        # Set FT vs FZ suffix and x_var accordingly
                        if [[ "$serial" -eq 24 ]]; then
                            prefix="fz"
                            add_cmd="--y_var_name ${y_var} --x_var_name ${x_var}"
                        else
                            prefix="ft"
                            if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" || "$metric" == "dist_intra_last_layer_train" || "$metric" == "dist_inter_last_layer_train" ]]; then
                                add_cmd="--y_var_name ${y_var} --x_var_name ${x_var}_ft"
                            elif [[ "$metric" == "acc_in1k" || "$metric" == "ada_ratio" ]]; then
                                add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
                            else
                                add_cmd="--y_var_name ${y_var} --x_var_name ${x_var}_ft"
                            fi
                        fi

                        output_file="${model_label}${prefix}_${y_var}_${metric,,}_${dataset}"
                        cmd="${base_cmd} --input_file ${input_path} ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
                        echo ""
                        echo "Running: ${cmd}"
                        eval "${cmd}"
                    done
                done
            done

            # -- Combined (ResNet + ViT) FT and FZ plots --
            for dataset in "${datasets_array[@]}"; do
                for serial in "${serials[@]}"; do
                    if [[ "$serial" -eq 24 ]]; then
                        model_used=${all_models_fz[*]}
                        prefix="fz"
                        if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" || "$metric" == "dist_intra_last_layer_train" || "$metric" == "dist_inter_last_layer_train" ]]; then
                            add_cmd="--y_var_name ${y_var} --x_var_name cka_last_${state}"
                        else
                            add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
                        fi
                    else
                        model_used=${all_models_ft[*]}
                        prefix="ft"
                        if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" || "$metric" == "dist_intra_last_layer_train" || "$metric" == "dist_inter_last_layer_train" ]]; then
                            add_cmd="--y_var_name ${y_var} --x_var_name cka_last_${state}_ft"
                        elif [[ "$metric" == "acc_in1k" || "$metric" == "ada_ratio" ]]; then
                            add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
                        else
                            add_cmd="--y_var_name ${y_var} --x_var_name ${metric}_ft"
                        fi
                    fi
                    output_file="both${prefix}_${y_var}_${metric,,}_${dataset}"
                    cmd="${base_cmd} --input_file ${input_path} ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
                    echo ""
                    echo "Running: ${cmd}"
                    eval "${cmd}"
                done
            done

            # -- Cross plots: acc_ft vs metric_fz --
            for dataset in "${datasets_array[@]}"; do
                for serial in "${serials[@]}"; do
                    if [[ "$serial" -eq 24 ]]; then
                        # Skip FZ serial for cross plots
                        continue
                    fi
                    model_used=${all_models_ft[*]}
                    if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" || "$metric" == "dist_intra_last_layer_train" || "$metric" == "dist_inter_last_layer_train" ]]; then
                        add_cmd="--y_var_name ${y_var} --x_var_name cka_last_${state}_matched"
                        alter_name="bothaccftvsmetricfz"
                    elif [[ "$metric" == "dist_intra_last_layer_train" || "$metric" == "dist_intra_last_layer_test" ]]; then
                        add_cmd="--y_var_name ${y_var} --x_var_name dist_intra_last_${state}_matched"
                        alter_name="bothaccftvsmetricfz"
                    elif [[ "$metric" == "dist_inter_last_layer_train" || "$metric" == "dist_inter_last_layer_test" ]]; then
                        add_cmd="--y_var_name ${y_var} --x_var_name dist_inter_last_${state}_matched"
                        alter_name="bothaccftvsmetricfz"
                    else
                        add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
                        alter_name="bothaccftvsmetricfz"
                    fi
                    output_file="${alter_name}_${y_var}_${metric,,}_${dataset}"
                    cmd="${base_cmd} --input_file ${input_path} ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
                    echo ""
                    echo "Running: ${cmd}"
                    eval "${cmd}"
                done
            done

            # -- Combined FT+FZ plots --
            for dataset in "${datasets_array[@]}"; do
                output_file="bothft+fz_${y_var}_${metric,,}_${dataset}"
                if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" ]]; then
                    add_cmd="--y_var_name ${y_var} --x_var_name cka_last_${state}_matched"
                elif [[ "$metric" == "dist_intra_last_layer_train" || "$metric" == "dist_intra_last_layer_test" ]]; then
                    add_cmd="--y_var_name ${y_var} --x_var_name dist_intra_last_${state}_matched"
                elif [[ "$metric" == "dist_inter_last_layer_train" || "$metric" == "dist_inter_last_layer_test" ]]; then
                    add_cmd="--y_var_name ${y_var} --x_var_name dist_inter_last_${state}_matched"
                elif [[ "$metric" == "acc_in1k" || "$metric" == "ada_ratio" ]]; then
                    add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
                else
                    add_cmd="--y_var_name ${y_var} --x_var_name ${metric}_matched"
                fi
                cmd="${base_cmd} --input_file ${input_path} ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serials[*]} --keep_methods ${all_models[*]} --output_file ${output_file}"
                echo ""
                echo "Running: ${cmd}"
                eval "${cmd}"
            done
        done
    done
done


# ============================================================
# -- Merged Correlation Plot Grids --
# ============================================================

# merge all regression plots into multiplot grids per accuracy metric
# accuracies=('ap_w')
# base_cmd="python merge_corr_plots.py"
for accuracy in "${accuracies[@]}"; do
    if [[ "$accuracy" == "val_acc_level1" ]]; then
        name="acc1_multiplot"; title_extend="Level 1 Accuracy"
    elif [[ "$accuracy" == "val_acc_level2" ]]; then
        name="acc2_multiplot"; title_extend="Level 2 Accuracy"
    else
        name="wap_multiplot"; title_extend="Weighted Accuracy Precision"
    fi
    cmd="${base_cmd} --input_folder results_all/new_plots/${accuracy} --output_folder results_all/new_plots/merged/${name} --output_file ${name} --title '${title_extend}' --x_filter rnfz rnft vitfz vitft bothft bothfz bothaccftvsmetricfz"
    echo ""
    echo "Running: ${cmd}"
    eval "${cmd}"
done


# ============================================================
# -- Top Model Performance Plots --
# ============================================================

# top best/worst model box plots (Top-1 and wAP)
# top3_best_models=('hiresnet50.in1k_mocov3' 'hiresnet50.in1k_swav' 'hiresnet50.in1k_spark' 'hiresnet50.a1_in1k' 'hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_siglip_224.v2_webli' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_clip_224.laion2b')

# top models top-1 accuracy
# base_cmd="python plot.py --y_label 'Accuracy(%)' --y_var_name val_acc_level1 --hue_var_name dataset_name --x_rotation 20 --x_var_name method --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance"
# output_file="top3_best_models_plot_top1"
# cmd="${base_cmd} --input_file data/hierarchical_all.csv --font_scale 1.75 --keep_serials 23 24 --keep_methods ${top3_best_models[*]} --output_file ${output_file} --title 'Top-2 Best & Worst-2 for Both Architecture (Top-1 Acc)' --keep_datasets ${datasets_array[*]}"
# echo "Running: ${cmd}"
# eval "${cmd}"

# top models weighted average precision
# base_cmd="python plot.py --y_label 'Weighted Average Precision(wAP, %)' --y_var_name ap_w --hue_var_name dataset_name --x_rotation 20 --x_var_name method --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance"
# output_file="top3_best_models_plot_ap_w"
# cmd="${base_cmd} --input_file data/hierarchical_all.csv --font_scale 1.75 --keep_serials 23 24 --keep_methods ${top3_best_models[*]} --output_file ${output_file} --title 'Top-2 Best & Worst-2 for Both Architecture (wAP)' --keep_datasets ${datasets_array[*]}"
# echo "Running: ${cmd}"
# eval "${cmd}"


# ============================================================
# -- Max Accuracy Variation Plots --
# ============================================================

# max accuracy variation across ResNet and ViT models (Top-1)
# base_cmd="python plot.py --y_label 'Accuracy(%)' --font_scale 1.75 --y_var_name acc_max --type_plot box --fig_size 10 6 --summarized --results_dir results_all/new_plots/dif_max_acc"
# cmd="${base_cmd} --input_file results_all/acc/summary_val_acc_level1_main.csv --keep_serials ${serials[*]} --x_label 'Datasets' --x_var_name dataset_name --hue_var_name serial --keep_methods ${all_resnet[*]} --keep_datasets ${datasets_array[*]} --output_file max_models_performance_top1 --title 'Max Accuracies Variations Between Resnet Models (Top1)'"
# echo "Running: ${cmd}"; eval "${cmd}"
# cmd="${base_cmd} --input_file results_all/acc/summary_val_acc_level1_main.csv --keep_serials ${serials[*]} --x_label 'Datasets' --x_var_name dataset_name --hue_var_name serial --keep_methods ${all_vit[*]} --keep_datasets ${datasets_array[*]} --output_file max_models_performance_vit_top1 --title 'Max Accuracies Variations Between ViT Models (Top1)'"
# echo "Running: ${cmd}"; eval "${cmd}"

# max accuracy variation across ResNet and ViT models (wAP)
# base_cmd="python plot.py --y_label 'Weighted Average Precision(wAP, %)' --font_scale 1.75 --y_var_name acc_max --type_plot box --fig_size 10 6 --summarized --results_dir results_all/new_plots/dif_max_acc"
# cmd="${base_cmd} --input_file results_all/acc/summary_ap_w_main.csv --keep_serials ${serials[*]} --x_label 'Datasets' --x_var_name dataset_name --hue_var_name serial --keep_methods ${all_resnet[*]} --keep_datasets ${datasets_array[*]} --output_file max_models_performance_ap_w --title 'Max Accuracies Variations Between Resnet Models (wAP)'"
# echo "Running: ${cmd}"; eval "${cmd}"
# cmd="${base_cmd} --input_file results_all/acc/summary_ap_w_main.csv --keep_serials ${serials[*]} --x_label 'Datasets' --x_var_name dataset_name --hue_var_name serial --keep_methods ${all_vit[*]} --keep_datasets ${datasets_array[*]} --output_file max_models_performance_vit_ap_w --title 'Max Accuracies Variations Between ViT Models (wAP)'"
# echo "Running: ${cmd}"; eval "${cmd}"
