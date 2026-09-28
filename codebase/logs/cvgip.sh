# # download
# python download_save_wandb_data.py --serials 23 24 --output_file hierarchical_stage1.csv
# python download_save_wandb_data_feature_metrics.py

# # merge data from acc, corr metrics and pretraining stats
# python merge_acc_metrics_stats.py

# # lr script
# # python lr_script.py

#  # accuracy
# python summarize_acc.py
# python summarize_acc_metrics.py

# # cost metrics including flops, no params, trainable params
# python summarize_cost.py
# python summarize_cost.py --acc_to_use val_acc_level2 --output_file cost_val_acc_level2
# python summarize_cost.py --acc_to_use ap_w --output_file cost_ap_w

datasets=('aircraft' 'cub' 'cars')
serials=('23' '24')

all_resnet_ft=('hiresnet50.tv_in1k' 'hiresnet50.tv2_in1k' 'hiresnet50.gluon_in1k' 'hiresnet50.fb_swsl_ig1b_ft_in1k' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k' 'hiresnet50.a1_in1k' 'hiresnet50.in1k_mocov3' 'hiresnet50.in1k_spark' 'hiresnet50.in1k_supcon' 'hiresnet50.in1k_swav' 'hiresnet50.in21k_miil')
all_vit_ft=('hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_224_miil.in21k' 'hideit_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_clip_224.laion2b' 'hivit_base_patch16_224.mae' 'hivit_base_patch16_224.in1k_mocov3' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_siglip_224.v2_webli')

all_resnet_fz=('hiresnet50.tv_in1k_fz' 'hiresnet50.tv2_in1k_fz' 'hiresnet50.gluon_in1k_fz' 'hiresnet50.fb_swsl_ig1b_ft_in1k_fz' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz' 'hiresnet50.a1_in1k_fz' 'hiresnet50.in1k_mocov3_fz' 'hiresnet50.in1k_spark_fz' 'hiresnet50.in1k_supcon_fz' 'hiresnet50.in1k_swav_fz' 'hiresnet50.in21k_miil_fz')
all_vit_fz=('hivit_base_patch16_224.orig_in21k_fz' 'hivit_base_patch16_224_miil.in21k_fz' 'hideit_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz' 'hivit_base_patch16_clip_224.laion2b_fz' 'hivit_base_patch16_224.mae_fz' 'hivit_base_patch16_224.in1k_mocov3_fz' 'hivit_base_patch16_224.dino_fz' 'hivit_base_patch16_siglip_224.v2_webli_fz')

all_resnet=('hiresnet50.tv_in1k' 'hiresnet50.tv2_in1k' 'hiresnet50.gluon_in1k' 'hiresnet50.fb_swsl_ig1b_ft_in1k' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k' 'hiresnet50.a1_in1k' 'hiresnet50.in1k_mocov3' 'hiresnet50.in1k_spark' 'hiresnet50.in1k_supcon' 'hiresnet50.in1k_swav' 'hiresnet50.in21k_miil' 'hiresnet50.tv_in1k_fz' 'hiresnet50.tv2_in1k_fz' 'hiresnet50.gluon_in1k_fz' 'hiresnet50.fb_swsl_ig1b_ft_in1k_fz' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz' 'hiresnet50.a1_in1k_fz' 'hiresnet50.in1k_mocov3_fz' 'hiresnet50.in1k_spark_fz' 'hiresnet50.in1k_supcon_fz' 'hiresnet50.in1k_swav_fz' 'hiresnet50.in21k_miil_fz')
all_vit=('hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_224_miil.in21k' 'hideit_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_clip_224.laion2b' 'hivit_base_patch16_224.mae' 'hivit_base_patch16_224.in1k_mocov3' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_siglip_224.v2_webli' 'hivit_base_patch16_224.orig_in21k_fz' 'hivit_base_patch16_224_miil.in21k_fz' 'hideit_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz' 'hivit_base_patch16_clip_224.laion2b_fz' 'hivit_base_patch16_224.mae_fz' 'hivit_base_patch16_224.in1k_mocov3_fz' 'hivit_base_patch16_224.dino_fz' 'hivit_base_patch16_siglip_224.v2_webli_fz')

input_path="data/hierarchical_stage1.csv"

architectures=("resnet" "vit")
for serial in "${serials[@]}"; do
    for architecture in "${architectures[@]}"; do
        if [[ "$architecture" == "resnet" ]]; then
            use_group="all_resnet"
            name_model="ResNet"
        else
            use_group="all_vit"
            name_model="ViT"
        fi
        
        if [[ "$serial" == 24 ]]; then
            suffix=fz
            name_suffix="FZ"
        else
            suffix=ft
            name_suffix="FT"
        fi
        output_file="distribution_accross_${use_group}_${suffix}"
        array_name="all_${architecture}[@]"
        methods=("${!array_name}")
        base_cmd="python plot.py --loc_legend 'lower right' --type_plot line --log_scale_x --title 'Mean wAP across ${name_model} ${name_suffix} Backbones vs LR for each Dataset' --fig_size 8 6 --input_file ${input_path} --results_dir results_all/cvgip/ --output_file ${output_file} --keep_methods ${methods[@]} --keep_serials ${serial} --x_label 'Learning Rate (LR)' --y_label 'Weighted Average Precision(wAP, %)' --keep_datasets "${datasets[*]}" --y_var_name ap_w --x_var_name lr --hue_var_name dataset_name"
        echo ""
        echo "Running: ${base_cmd}"
        eval "${base_cmd}"
    done
done

for model in "${all_resnet_ft[@]}"; do
    suffix=ft
    output_file="${model}_${suffix}"
    # array_name="all_${architecture}_${suffix}[@]"
    # methods=("${!array_name}")
    base_cmd="python plot.py --type_plot line --loc_legend 'lower right' --log_scale_x --title 'wAP for ${model} FT Backbone vs LR for each Dataset' --fig_size 8 6 --input_file ${input_path} --results_dir results_all/cvgip/rnft --output_file ${output_file} --keep_methods ${model} --keep_serials 23 --x_label 'Learning Rate (LR)' --y_label 'Weighted Average Precision(wAP, %)' --keep_datasets "${datasets[*]}" --y_var_name ap_w --x_var_name lr --hue_var_name dataset_name"
    echo ""
    echo "Running: ${base_cmd}"
    eval "${base_cmd}"
done

for model in "${all_resnet_fz[@]}"; do
    suffix=fz
    output_file="${model}"
    # array_name="all_${architecture}_${suffix}[@]"
    # methods=("${!array_name}")
    base_cmd="python plot.py --type_plot line --loc_legend 'lower right' --log_scale_x --title 'wAP for ${model} FZ Backbone vs LR for each Dataset' --fig_size 8 6 --input_file ${input_path} --results_dir results_all/cvgip/rnfz --output_file ${output_file} --keep_methods ${model} --keep_serials 24 --x_label 'Learning Rate (LR)' --y_label 'Weighted Average Precision(wAP, %)' --keep_datasets "${datasets[*]}" --y_var_name ap_w --x_var_name lr --hue_var_name dataset_name"
    echo ""
    echo "Running: ${base_cmd}"
    eval "${base_cmd}"
done

for model in "${all_vit_ft[@]}"; do
    suffix=ft
    output_file="${model}_${suffix}"
    # array_name="all_${architecture}_${suffix}[@]"
    # methods=("${!array_name}")
    base_cmd="python plot.py --type_plot line --loc_legend 'lower right' --log_scale_x --title 'wAP for ${model} FT Backbone vs LR for each Dataset' --fig_size 8 6 --input_file ${input_path} --results_dir results_all/cvgip/vitft --output_file ${output_file} --keep_methods ${model} --keep_serials 23 --x_label 'Learning Rate (LR)' --y_label 'Weighted Average Precision(wAP, %)' --keep_datasets "${datasets[*]}" --y_var_name ap_w --x_var_name lr --hue_var_name dataset_name"
    echo ""
    echo "Running: ${base_cmd}"
    eval "${base_cmd}"
done

for model in "${all_vit_fz[@]}"; do
    suffix=fz
    output_file="${model}"
    # array_name="all_${architecture}_${suffix}[@]"
    # methods=("${!array_name}")
    base_cmd="python plot.py --type_plot line --loc_legend 'lower right' --log_scale_x --title 'wAP for ${model} FZ Backbone vs LR for each Dataset' --fig_size 8 6 --input_file ${input_path} --results_dir results_all/cvgip/vitfz --output_file ${output_file} --keep_methods ${model} --keep_serials 24 --x_label 'Learning Rate (LR)' --y_label 'Weighted Average Precision(wAP, %)' --keep_datasets "${datasets[*]}" --y_var_name ap_w --x_var_name lr --hue_var_name dataset_name"
    echo ""
    echo "Running: ${base_cmd}"
    eval "${base_cmd}"
done