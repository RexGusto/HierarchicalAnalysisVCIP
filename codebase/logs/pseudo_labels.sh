#!/bin/bash

# ============================================================
# Model Arrays
# ============================================================

# ResNet and ViT backbone lists
all_resnet=('hiresnet50.tv_in1k' 'hiresnet50.tv2_in1k' 'hiresnet50.gluon_in1k' 'hiresnet50.fb_swsl_ig1b_ft_in1k' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k' 'hiresnet50.a1_in1k' 'hiresnet50.in1k_mocov3' 'hiresnet50.in1k_spark' 'hiresnet50.in1k_supcon' 'hiresnet50.in1k_swav' 'hiresnet50.in21k_miil')
all_vit=('hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_224_miil.in21k' 'hideit_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_clip_224.laion2b' 'hivit_base_patch16_224.mae' 'hivit_base_patch16_224.in1k_mocov3' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_siglip_224.v2_webli')
all_models=("${all_resnet[@]}" "${all_vit[@]}")


# ============================================================
# -- Hyperparameter Sweeps (Pseudo) --
# ============================================================

# -- Epochs --
# python download_save_wandb_data.py --serials 64 65 --output_file hierarchical_epochs.csv
# python summarize_acc.py --input_file data/hierarchical_epochs.csv --main_serials 64 65 --results_dir results_all/acc/acc_epochs
# python compute_diff.py --input_csv results_all/acc/acc_epochs/summary_val_acc_level1_main.csv --main_serials 64 65 --output_dir results_all/pseudo_hyperparam/epochs

# -- Image Size --
# python download_save_wandb_data.py --serials 66 67 --output_file hierarchical_is.csv
# python summarize_acc.py --input_file data/hierarchical_is.csv --main_serials 66 67 --results_dir results_all/acc/acc_is
# python compute_diff.py --input_csv results_all/acc/acc_is/summary_val_acc_level1_main.csv --main_serials 66 67 --output_dir results_all/pseudo_hyperparam/image_size

# -- Augmentations --
# python download_save_wandb_data.py --serials 68 69 --output_file hierarchical_augs.csv
# python summarize_acc.py --input_file data/hierarchical_augs.csv --main_serials 68 69 --results_dir results_all/acc/acc_augs
# python compute_diff.py --input_csv results_all/acc/acc_augs/summary_val_acc_level1_main.csv --main_serials 68 69 --output_dir results_all/pseudo_hyperparam/augs

# -- Model Size --
# python download_save_wandb_data.py --serials 70 --output_file hierarchical_model_size.csv
# python summarize_acc.py --input_file data/hierarchical_model_size.csv --main_serials 70 --results_dir results_all/acc/acc_model_size
# python compute_diff.py --input_csv results_all/acc/acc_model_size/summary_val_acc_level1_main.csv --main_serials 70 --output_dir results_all/pseudo_hyperparam/model_size

# -- Clustering Algorithm --
# python download_save_wandb_data.py --serials 40 52 53 --output_file hierarchical_algo.csv
# python summarize_acc.py --input_file data/hierarchical_algo.csv --main_serials 40 52 53 --results_dir results_all/acc/acc_algo
# python compute_diff.py --input_csv results_all/acc/acc_algo/summary_val_acc_level1_main.csv --main_serials 40 52 53 --output_dir results_all/pseudo_hyperparam/clus_algo

# -- Different Feature Extractor --
# python download_save_wandb_data.py --serials 40 51 --output_file hierarchical_diff_extractor.csv
# python summarize_acc.py --input_file data/hierarchical_diff_extractor.csv --main_serials 40 51 --results_dir results_all/acc/acc_extractor
# python compute_diff.py --input_csv results_all/acc/acc_extractor/summary_val_acc_level1_main.csv --main_serials 40 51 --output_dir results_all/pseudo_hyperparam/diff_extractor

# -- Main Results (Baseline / Real / Pseudo) --
# python download_save_wandb_data.py --serials 40 23 32 --output_file hierarchical_pseudo_2.csv
# python summarize_acc.py --input_file data/hierarchical_pseudo_2.csv --main_serials 40 23 32 --results_dir results_all/acc/acc_pseudo_main
# python compute_diff.py --input_csv results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --main_serials 32 23 40 --output_dir results_all/pseudo_hyperparam/main_results


# ============================================================
# -- Soylocal Runs --
# ============================================================

# Download soylocal wandb data (commented out after first run)
# python download_save_wandb_data.py --serials 54 55 --output_file hierarchical_soylocal.csv

# Summarize accuracy and compute diff for soylocal
# python summarize_acc.py --input_file data/hierarchical_soylocal.csv --main_serials 55 54 --results_dir results_all/acc/acc_soylocal
# python compute_diff.py --input_csv results_all/acc/acc_soylocal/summary_val_acc_level1_main.csv --main_serials 55 54 --output_dir results_all/pseudo_hyperparam/soylocal

# Plot Resnet backbones on Soylocal
# python -u plot.py --keep_serials 55 54 --fig_size 10 6 --loc_legend 'lower right' --font_scale 1.2 --keep_dataset soylocal --x_rotation 20 --keep_methods "${all_resnet[@]}" --x_var_name method --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name serial --type_plot box --input_file data/hierarchical_soylocal.csv --output_file compare_resnet_soylocal --title 'Accuracy of Resnet backbones on Soylocal dataset'

# Plot ViT backbones on Soylocal
# python -u plot.py --keep_serials 55 54 --fig_size 10 6 --loc_legend 'lower right' --font_scale 1.2 --keep_dataset soylocal --x_rotation 20 --keep_methods "${all_vit[@]}" --x_var_name method --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name serial --type_plot box --input_file data/hierarchical_soylocal.csv --output_file compare_vit_soylocal --title 'Accuracy of ViT backbones on Soylocal dataset'

# original soylocal backbone comparison (serials 39 32)
# python -u plot.py --keep_serials 39 32 --font_scale 1.75 --fig_size 8 6 --loc_legend 'lower right' --keep_dataset soylocal --x_var_name n_cluster_ratio --y_var_name val_acc_level1 --x_label 'Cluster Ratio (%)' --y_label 'Accuracy (%)' --hue_var_name serial --type_plot box --input_file data/hierarchical_soylocal.csv --output_file compare_soylocal --title 'Accuracy of backbones on Soylocal dataset'

# soylocal ADA ratio (serials 39 32)
# python -u plot.py --keep_serials 39 32 --font_scale 1.75 --fig_size 8 6 --summarized --keep_dataset soylocal --x_var_name n_cluster_ratio --y_var_name ada_ratio --x_label 'Cluster Ratio (%)' --y_label 'Coefficient of Variation (%)' --hue_var_name serial --type_plot box --input_file results_all/acc/acc_soylocal/summary_val_acc_level1_main.csv --output_file compare_soylocal_ada --title 'Overall performance of different models on Soylocal dataset'

# soylocal heatmaps
# python -u plot.py --keep_serials 39 32 --font_scale 1.5 --fig_size 8 6 --keep_dataset soylocal --x_var_name n_cluster_ratio --y_var_name lr --x_label 'Cluster Ratio (%)' --y_label 'Learning Rates' --hue_var_name val_acc_level1 --input_file data/hierarchical_soylocal.csv --output_file heatmap_soylocal --title 'Heatmap of Soylocal dataset' --type_plot heatmap
# python -u plot.py --keep_serials 39 32 --font_scale 1.5 --fig_size 8 6 --keep_dataset soylocal --x_var_name lr --y_var_name batch_size --x_label 'Learning Rates' --y_label 'Batch Size' --hue_var_name val_acc_level1 --input_file data/hierarchical_soylocal.csv --output_file heatmap_soylocal_1d --title 'Heatmap of Soylocal dataset' --type_plot heatmap


# ============================================================
# -- Other Soy Datasets (cotton, soyageing, soygene, etc.) --
# ============================================================

# Download other soy datasets (commented after first run)
# python download_save_wandb_data.py --serials 83 86 --output_file other_soy_stuff.csv

# Summarize and compute diff for other soy datasets
# python summarize_acc.py --input_file data/other_soy_stuff.csv --main_serials 83 86 --results_dir results_all/acc/acc_othersoy
# python compute_diff.py --input_csv results_all/acc/acc_othersoy/summary_val_acc_level1_main.csv --main_serials 83 86 --output_dir results_all/pseudo_hyperparam/othersoy

# cotton seed heatmaps (ViT and DeiT)
# python -u plot.py --keep_serials 84 85 --keep_method hivit_base_patch16_224.orig_in21k --font_scale 1.5 --fig_size 8 6 --keep_dataset cotton --x_var_name n_cluster_ratio --y_var_name seed --y_label 'Seeds' --x_label 'Cluster Ratio (%)' --hue_var_name val_acc_level1 --input_file data/cotton_seeds.csv --output_file heatmap_cotton_seed_vit --title 'Heatmap of Cotton with different seeds on ViT' --type_plot heatmap
# python -u plot.py --keep_serials 84 85 --keep_method hideit_base_patch16_224.fb_in1k --font_scale 1.5 --fig_size 8 6 --keep_dataset cotton --x_var_name n_cluster_ratio --y_var_name seed --y_label 'Seeds' --x_label 'Cluster Ratio (%)' --hue_var_name val_acc_level1 --input_file data/cotton_seeds.csv --output_file heatmap_cotton_seed_deit --title 'Heatmap of Cotton with different seeds on Deit' --type_plot heatmap

# per-dataset box plots for all leaves datasets
# leaves_datasets=('cotton' 'soyageing_r1' 'soygene' 'soyglobal' 'soyageing')
# for dataset in ${leaves_datasets[@]}; do
#     python -u plot.py --keep_serials 83 86 --keep_method hivit_base_patch16_224.orig_in21k --font_scale 1.5 --fig_size 8 6 --keep_dataset ${dataset} --x_var_name n_cluster_ratio --y_var_name val_acc_level1 --x_label 'Cluster Ratio (%)' --input_file data/other_soy_stuff.csv --results_dir results_all/plots/other_leaves --output_file box_${dataset}_vit --title "${dataset} on ViT" --type_plot box
#     python -u plot.py --keep_serials 83 86 --keep_method hideit_base_patch16_224.fb_in1k --font_scale 1.5 --fig_size 8 6 --keep_dataset ${dataset} --x_var_name n_cluster_ratio --y_var_name val_acc_level1 --x_label 'Cluster Ratio (%)' --input_file data/other_soy_stuff.csv --results_dir results_all/plots/other_leaves --output_file box_${dataset}_deit --title "${dataset} on Deit" --type_plot box
# done

# LR scripts for other leaves (pseudo and hier)
# python -u lr_script.py --input_file data/other_leaves.csv --prefix "python -u tools/train.py --serial 84" --output_file other_leaves_pseudo --use_ratios
# python -u lr_script.py --input_file data/other_leaves.csv --prefix "python -u tools/train.py --serial 85" --output_file hier_all


# ============================================================
# -- Hyperparameter Sweep Plots (Pseudo) --
# ============================================================

# augmentation comparison box plot
# python -u plot.py --keep_serials 68 69 --x_var_name serial --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_augs.csv --output_file compare_augs --title 'Comparison of 12 augmentations of Aircraft with hideit3_base_patch16_224.fb_in1k model in Baseline vs Pseudo-Hierarchy settings'

# image size comparison box plot
# python -u plot.py --keep_serials 66 67 --x_var_name serial --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_is.csv --output_file compare_is --title 'Comparison of 10 image sizes (is) of Aircraft with hideit3_base_patch16_224.fb_in1k model in Baseline vs Pseudo-Hierarchy settings'

# epoch comparison box plot
# python -u plot.py --keep_serials 64 65 --x_var_name serial --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_epochs.csv --output_file compare_epochs --title 'Comparison of 10 epochs of Aircraft with hideit3_base_patch16_224.fb_in1k model in Baseline vs Pseudo-Hierarchy settings'

# model size comparison box plot
# python -u plot.py --keep_serials 70 --x_var_name model_name_extractor --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_model_size.csv --output_file compare_model_size --title 'Comparison of hideit3_base_patch16_224.fb_in1k model sizes on Aircraft in Baseline vs Pseudo-Hierarchy settings'

# model size ADA ratio bar plot
# python -u plot.py --keep_serials 70 --summarized --x_var_name model_name_extractor --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --type_plot bar --input_file results_all/acc/acc_model_size/summary_val_acc_level1_main.csv --output_file compare_model_size_ada --title 'Comparison of hideit3_base_patch16_224.fb_in1k model sizes on Aircraft in Baseline vs Pseudo-Hierarchy settings'

# model size ADA ratio bar (deit3 sizes only)
# python -u plot.py --keep_serials 70 --font_scale 1.5 --fig_size 10 6 --summarized --keep_extractor 'hideit3_base_patch16_224.fb_in1k' 'hideit3_large_patch16_224.fb_in1k' 'hideit3_huge_patch16_224.fb_in1k' --x_var_name model_name_extractor --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --type_plot bar --input_file results_all/acc/acc_model_size/summary_val_acc_level1_main.csv --output_file compare_model_size_ada_alt --title 'hideit3_base_patch16_224.fb_in1k sizes on Aircraft in Pseudo-Hierarchy'

# clustering algorithm comparisons
# python -u plot.py --keep_serials 40 52 --x_ticks_labels 'HPH' 'Kmean' --font_scale 1.75 --fig_size 8 6 --keep_ratios 50 --keep_dataset aircraft_pl --keep_methods hideit3_base_patch16_224.fb_in1k --x_var_name serial --y_label 'Accuracy (%)' --y_var_name val_acc_level1 --type_plot box --input_file data/hierarchical_algo.csv --output_file compare_algo_all --title 'Clustering algorithms on Aircraft with hideit3_base_patch16_224.fb_in1k'
# python -u plot.py --keep_serials 40 52 --keep_ratios 50 --keep_dataset aircraft_pl --keep_methods hideit3_base_patch16_224.fb_in1k --x_var_name serial --y_label 'Accuracy (%)' --y_var_name val_acc_level1 --type_plot box --input_file data/hierarchical_algo.csv --output_file compare_algo_only_kmeans --title 'Comparison of different clustering algorithm on Aircraft with hideit3_base_patch16_224.fb_in1k model'

# extractor comparison (ResNet)
# python -u plot.py --keep_serials 40 51 --keep_ratios 50 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --keep_dataset aircraft_pl --keep_methods hideit3_base_patch16_224.fb_in1k --keep_extractor ${all_resnet[@]} --x_var_name model_name_extractor --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_diff_extractor.csv --output_file compare_extractors_rn --title 'Overall performance of different Resnet extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model'

# extractor comparison (ViT)
# python -u plot.py --keep_serials 40 51 --keep_ratios 50 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --keep_dataset aircraft_pl --keep_methods hideit3_base_patch16_224.fb_in1k --keep_extractor ${all_vit[@]} --x_var_name model_name_extractor --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --type_plot box --input_file data/hierarchical_diff_extractor.csv --output_file compare_extractors_vit --title 'Overall performance of ViT extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model'

# extractor StD and ADA ratio bars (ViT)
# python -u plot.py --keep_serials 40 51 --keep_ratios 50 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --summarized --keep_methods hideit3_base_patch16_224.fb_in1k --keep_dataset aircraft_pl --x_var_name model_name_extractor --y_var_name acc_std --y_label 'Accuracy Standard Deviation' --keep_extractor ${all_vit[@]} --x_var_name model_name_extractor --type_plot bar --input_file results_all/acc/acc_extractor/summary_val_acc_level1_main.csv --output_file compare_extractors_std_vit --title 'StD performance of ViT extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model'
# python -u plot.py --keep_serials 40 51 --keep_ratios 50 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --summarized --keep_methods hideit3_base_patch16_224.fb_in1k --keep_dataset aircraft_pl --x_var_name model_name_extractor --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --keep_extractor ${all_vit[@]} --x_var_name model_name_extractor --type_plot bar --input_file results_all/acc/acc_extractor/summary_val_acc_level1_main.csv --output_file compare_extractors_ada_vit --title 'ADA ratio of ViT extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model'

# extractor StD and ADA ratio bars (ResNet)
# python -u plot.py --keep_serials 40 51 --keep_ratios 50 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --summarized --keep_methods hideit3_base_patch16_224.fb_in1k --keep_dataset aircraft_pl --x_var_name model_name_extractor --y_var_name acc_std --y_label 'Accuracy Standard Deviation' --keep_extractor ${all_resnet[@]} --x_var_name model_name_extractor --type_plot bar --input_file results_all/acc/acc_extractor/summary_val_acc_level1_main.csv --output_file compare_extractors_std_rn --title 'StD performance of Resnet extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model'
# python -u plot.py --keep_serials 40 51 --keep_ratios 50 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --summarized --keep_methods hideit3_base_patch16_224.fb_in1k --keep_dataset aircraft_pl --x_var_name model_name_extractor --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --keep_extractor ${all_resnet[@]} --x_var_name model_name_extractor --type_plot bar --input_file results_all/acc/acc_extractor/summary_val_acc_level1_main.csv --output_file compare_extractors_ada_rn --title 'ADA ratio of Resnet extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model'

# all extractor ADA ratios combined
# python -u plot.py --keep_serials 40 51 --keep_ratios 50 --fig_size 24 6 --x_rotation 20 --font_scale 1.75 --summarized --keep_methods hideit3_base_patch16_224.fb_in1k --keep_dataset aircraft_pl --x_var_name model_name_extractor --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --keep_extractor ${all_models[@]} --x_var_name model_name_extractor --type_plot bar --input_file results_all/acc/acc_extractor/summary_val_acc_level1_main.csv --output_file compare_extractors_ada_all --title 'CV of all extractors on Aircraft with hideit3_base_patch16_224.fb_in1k model'


# ============================================================
# -- Main Results Plots (3 Settings: Baseline / Real / Pseudo) --
# ============================================================

# accuracy and ADA ratio across 3 datasets, by serial
# python -u plot.py --keep_serials 32 23 40 --loc_legend 'lower right' --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name serial --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name dataset_name --type_plot box --input_file data/hierarchical_pseudo_2.csv --output_file compare_main_plot_acc --title 'Overall performance of different models on 3 settings: Baseline, Real Hierarchy, Pseudo-Hierarchy'
# python -u plot.py --keep_serials 32 23 40 --keep_dataset aircraft cub cars --summarized --aggregate_dataset --x_var_name serial --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --hue_var_name dataset_name --type_plot box --input_file results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --output_file compare_main_plot_ada --title 'CV of different models on 3 settings: Baseline, Real Hierarchy, Pseudo-Hierarchy'

# accuracy and ADA by cluster ratio
# python -u plot.py --keep_serials 40 --keep_ratios 25 50 70 --keep_dataset aircraft cub cars --summarized --aggregate_dataset --x_var_name n_cluster_ratio --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --hue_var_name dataset_name --type_plot box --input_file results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --output_file compare_main_plot_ratios_ada --title 'CV of different pseudo-hierarchy ratios on 3 datasets'
# python -u plot.py --keep_serials 40 --loc_legend 'lower right' --keep_ratios 25 50 70 --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name n_cluster_ratio --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name dataset_name --type_plot box --input_file data/hierarchical_pseudo_2.csv --output_file compare_main_plot_ratios_acc --title 'Overall performance of different pseudo-hierarchy ratios on 3 datasets'

# alternative grouping by dataset_name (x-axis)
# python -u plot.py --keep_serials 32 23 40 --keep_ratios 0 50 --loc_legend 'lower right' --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name dataset_name --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name serial --type_plot box --input_file data/hierarchical_pseudo_2.csv --output_file compare_main_plot_acc_alt --title 'Overall performance of different models on 3 settings: Baseline, Real Hierarchy, Pseudo-Hierarchy'
# python -u plot.py --keep_serials 32 23 40 --keep_ratios 0 50 --keep_dataset aircraft cub cars --summarized --aggregate_dataset --x_var_name dataset_name --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --hue_var_name serial --type_plot box --input_file results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --output_file compare_main_plot_ada_alt --title 'CV of different models on 3 settings: Baseline, Real Hierarchy, Pseudo-Hierarchy'

# ratio comparison with pastel palette
# python -u plot.py --keep_serials 40 --keep_ratios 25 50 70 --keep_dataset aircraft cub cars --summarized --aggregate_dataset --x_var_name dataset_name --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --hue_var_name n_cluster_ratio --palette pastel --type_plot box --input_file results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --output_file compare_main_plot_ratios_ada_alt --title 'CV of different pseudo-hierarchy ratios on 3 datasets'
# python -u plot.py --keep_serials 40 --loc_legend 'lower right' --keep_ratios 25 50 70 --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name dataset_name --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name n_cluster_ratio --palette pastel --type_plot box --input_file data/hierarchical_pseudo_2.csv --output_file compare_main_plot_ratios_acc_alt --title 'Overall performance of different pseudo-hierarchy ratios on 3 datasets'

# per-dataset per-ratio backbone plots (nested loop)
# keep_ratios=('25' '50' '70')
datasets=('aircraft' 'cars' 'cub')
# for dataset in ${datasets[@]}; do
#     for ratio in ${keep_ratios[@]}; do
#         python -u plot.py --keep_serials 32 23 40 --font_scale 1.75 --fig_size 13 8 --x_rotation 20 --keep_ratios 0 ${ratio} --loc_legend 'lower left' --keep_methods ${all_resnet[@]} --keep_dataset ${dataset} --aggregate_dataset --x_var_name method --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name serial --type_plot box --input_file data/hierarchical_pseudo_2.csv --output_file compare_main_plot_acc_${dataset}_${ratio}_rn --title "Accuracy vs ResNet Backbones for ${dataset}"
#         python -u plot.py --keep_serials 32 23 40 --font_scale 1.75 --fig_size 13 8 --x_rotation 20 --keep_ratios 0 ${ratio} --loc_legend 'lower left' --keep_methods ${all_vit[@]} --keep_dataset ${dataset} --aggregate_dataset --x_var_name method --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name serial --type_plot box --input_file data/hierarchical_pseudo_2.csv --output_file compare_main_plot_acc_${dataset}_${ratio}_vit --title "Accuracy vs ViT Backbones for ${dataset}"
#         # python -u plot.py --keep_serials 32 23 40 --fig_size 20 6 --keep_ratios 0 50 --keep_dataset ${dataset} --keep_methods ${all_models[@]} --summarized --aggregate_dataset --x_var_name method --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --hue_var_name serial --type_plot box --input_file results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --output_file compare_main_plot_ada_${dataset} --title 'CV of different models on 3 settings: Baseline, Real Hierarchy, Pseudo-Hierarchy'
#         python -u plot.py --keep_serials 40 --font_scale 1.75 --fig_size 8 6 --palette pastel --keep_ratios 25 50 70 --keep_dataset ${dataset} --summarized --aggregate_dataset --x_var_name dataset_name --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --hue_var_name n_cluster_ratio --type_plot box --input_file results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --output_file compare_main_plot_ratios_ada_${dataset} --title "${dataset} Cluster Ratio Comparison"
#         python -u plot.py --keep_serials 40 --font_scale 1.75 --fig_size 8 6 --palette pastel --loc_legend 'lower right' --keep_ratios 25 50 70 --keep_dataset ${dataset} --aggregate_dataset --x_var_name dataset_name --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name n_cluster_ratio --type_plot box --input_file data/hierarchical_pseudo_2.csv --output_file compare_main_plot_ratios_acc_${dataset} --title "${dataset} Cluster Ratio Comparison"
#     done
# done

# # per-ratio ResNet and ViT cross-dataset plots
# for ratio in ${keep_ratios[@]}; do
#     python -u plot.py --keep_serials 40 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --loc_legend 'lower right' --keep_ratios ${ratio} --keep_extractor ${all_resnet[@]} --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name method --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name dataset_name --type_plot box --input_file data/hierarchical_pseudo_2.csv --output_file compare_main_plot_${ratio}_acc_rn --title "Overall performance of different pseudo-hierarchy with ${ratio}% ratios on 3 datasets"
#     python -u plot.py --keep_serials 40 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --keep_ratios ${ratio} --keep_dataset aircraft cub cars --keep_extractor ${all_resnet[@]} --summarized --aggregate_dataset --x_var_name method --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --hue_var_name dataset_name --type_plot bar --input_file results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --output_file compare_main_plot_${ratio}_ada_rn --title "CV of different pseudo-hierarchy with ${ratio}% ratios on 3 datasets"
#     python -u plot.py --keep_serials 40 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --loc_legend 'lower right' --keep_ratios ${ratio} --keep_extractor ${all_vit[@]} --keep_dataset aircraft cub cars --aggregate_dataset --x_var_name method --y_var_name val_acc_level1 --y_label 'Accuracy (%)' --hue_var_name dataset_name --type_plot box --input_file data/hierarchical_pseudo_2.csv --output_file compare_main_plot_${ratio}_acc_vit --title "Overall performance of different pseudo-hierarchy with ${ratio}% ratios on 3 datasets"
#     python -u plot.py --keep_serials 40 --fig_size 10 6 --x_rotation 20 --font_scale 0.75 --keep_ratios ${ratio} --keep_dataset aircraft cub cars --keep_extractor ${all_vit[@]} --summarized --aggregate_dataset --x_var_name method --y_var_name ada_ratio --y_label 'Coefficient of Variation (%)' --hue_var_name dataset_name --type_plot bar --input_file results_all/acc/acc_pseudo_main/summary_val_acc_level1_main.csv --output_file compare_main_plot_${ratio}_ada_vit --title "CV of different pseudo-hierarchy with ${ratio}% ratios on 3 datasets"
# done

# ============================================================
# -- CutMix and Hier. Label Smoothing --
# ============================================================
# python download_save_wandb_data.py --serials 83 86 54 55 90 91 92 93 94 95 --output_file label_smooth_cutmix.csv

# python -u plot.py --keep_serials 54 55 90 91 92 93 94 95 --keep_method hivit_base_patch16_224.orig_in21k --font_scale 1.5 --fig_size 10 6 --x_rotation 25 --keep_dataset soylocal --x_var_name serial --y_var_name val_acc_level1 --input_file data/label_smooth_cutmix.csv --results_dir results_all/plots/label_smooth_cutmix --output_file box_soylocal_vit --title "Soylocal on ViT" --type_plot box
# python -u plot.py --keep_serials 54 55 90 91 92 93 94 95 --keep_method hideit_base_patch16_224.fb_in1k --font_scale 1.5 --fig_size 10 6 --x_rotation 25 --keep_dataset soylocal --x_var_name serial --y_var_name val_acc_level1 --input_file data/label_smooth_cutmix.csv --results_dir results_all/plots/label_smooth_cutmix --output_file box_soylocal_deit --title "Soylocal on Deit" --type_plot box

# python -u plot.py --keep_serials 83 86 90 91 92 93 94 95 --keep_method hivit_base_patch16_224.orig_in21k --font_scale 1.5 --fig_size 10 6 --x_rotation 25 --keep_dataset cotton --x_var_name serial --y_var_name val_acc_level1 --input_file data/label_smooth_cutmix.csv --results_dir results_all/plots/label_smooth_cutmix --output_file box_cotton_vit --title "Cotton on ViT" --type_plot box
# python -u plot.py --keep_serials 83 86 90 91 92 93 94 95 --keep_method hideit_base_patch16_224.fb_in1k --font_scale 1.5 --fig_size 10 6 --x_rotation 25 --keep_dataset cotton --x_var_name serial --y_var_name val_acc_level1 --input_file data/label_smooth_cutmix.csv --results_dir results_all/plots/label_smooth_cutmix --output_file box_cotton_deit --title "Cotton on Deit" --type_plot box

# ============================================================
# -- 1-example model runs --
# ============================================================
# python download_save_wandb_data.py --serials 102 103 --output_file model_low_examples.csv

for dataset in ${datasets[@]}; do
    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 1 --keep_method ${all_resnet[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name val_acc_level1 --input_file data/model_low_examples.csv --results_dir results_all/plots/model_low_examples --output_file box_1_example_${dataset}_resnet --title "1-Example Resnet Models on ${dataset}" --type_plot box
    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 1 --keep_method ${all_vit[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name val_acc_level1 --input_file data/model_low_examples.csv --results_dir results_all/plots/model_low_examples --output_file box_1_example_${dataset}_vit --title "1-Example ViT Models on ${dataset}" --type_plot box

    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 3 --keep_method ${all_resnet[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name val_acc_level1 --input_file data/model_low_examples.csv --results_dir results_all/plots/model_low_examples --output_file box_3_example_${dataset}_resnet --title "3-Example Resnet Models on ${dataset}" --type_plot box
    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 3 --keep_method ${all_vit[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name val_acc_level1 --input_file data/model_low_examples.csv --results_dir results_all/plots/model_low_examples --output_file box_3_example_${dataset}_vit --title "3-Example ViT Models on ${dataset}" --type_plot box

    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 5 --keep_method ${all_resnet[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name val_acc_level1 --input_file data/model_low_examples.csv --results_dir results_all/plots/model_low_examples --output_file box_5_example_${dataset}_resnet --title "5-Example Resnet Models on ${dataset}" --type_plot box
    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 5 --keep_method ${all_vit[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name val_acc_level1 --input_file data/model_low_examples.csv --results_dir results_all/plots/model_low_examples --output_file box_5_example_${dataset}_vit --title "5-Example ViT Models on ${dataset}" --type_plot box
done

# python summarize_acc.py --input_file data/model_low_examples.csv --main_serials 102 103 --results_dir results_all/acc/acc_few_examples

for dataset in ${datasets[@]}; do
    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 1 --keep_method ${all_resnet[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name ada_ratio --input_file results_all/acc/acc_few_examples/summary_val_acc_level1_main.csv --results_dir results_all/plots/model_low_examples --output_file bar_1_example_${dataset}_resnet --title "1-Example Resnet Models on ${dataset}" --type_plot bar --summarized
    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 1 --keep_method ${all_vit[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name ada_ratio --input_file results_all/acc/acc_few_examples/summary_val_acc_level1_main.csv --results_dir results_all/plots/model_low_examples --output_file bar_1_example_${dataset}_vit --title "1-Example ViT Models on ${dataset}" --type_plot bar --summarized

    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 3 --keep_method ${all_resnet[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name ada_ratio --input_file results_all/acc/acc_few_examples/summary_val_acc_level1_main.csv --results_dir results_all/plots/model_low_examples --output_file bar_3_example_${dataset}_resnet --title "3-Example Resnet Models on ${dataset}" --type_plot bar --summarized
    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 3 --keep_method ${all_vit[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name ada_ratio --input_file results_all/acc/acc_few_examples/summary_val_acc_level1_main.csv --results_dir results_all/plots/model_low_examples --output_file bar_3_example_${dataset}_vit --title "3-Example ViT Models on ${dataset}" --type_plot bar --summarized

    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 5 --keep_method ${all_resnet[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name ada_ratio --input_file results_all/acc/acc_few_examples/summary_val_acc_level1_main.csv --results_dir results_all/plots/model_low_examples --output_file bar_5_example_${dataset}_resnet --title "5-Example Resnet Models on ${dataset}" --type_plot bar --summarized
    python -u plot.py --font_scale 1.2 --fig_size 12 6 --x_rotation 25 --keep_ipc 5 --keep_method ${all_vit[@]} --keep_dataset ${dataset} ${dataset}_baseline --hue_var_name serial --x_var_name method --y_var_name ada_ratio --input_file results_all/acc/acc_few_examples/summary_val_acc_level1_main.csv --results_dir results_all/plots/model_low_examples --output_file bar_5_example_${dataset}_vit --title "5-Example ViT Models on ${dataset}" --type_plot bar --summarized
done

# python -u plot.py --font_scale 1.2 --fig_size 12 6 --keep_dataset aircraft aircraft_baseline --x_var_name method --y_var_name val_acc_level1 --hue_var_name img_per_class --input_file data/model_low_examples.csv --results_dir results_all/plots/model_low_examples --output_file box_1_example_aircraft_compare --title "1-Example Baseline vs Hierarchy on Aircraft" --type_plot box
