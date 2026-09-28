# Provenance of the VCIP26 numbers

Every number below is printed in `drafts/VCIP26/latex/camera_ready.tex`.
`codebase.verify.verify_paper_numbers` re-derives each one from its source and reports
whether it matches.

- **Maximum.** The highest `val_acc_level1` (top-1 accuracy on the evaluation split) over
  all runs of a backbone in the given serial. For serial 54 this covers all 10 learning rates
  and all cluster ratios. For serial 86 it covers 10 learning rates and ratios 70 and 90.
- **Gain.** A difference computed from the two printed, rounded operands.
- **Rounding.** Half-up, to the printed number of decimals.

| Printed number | Source | Filter | Aggregation |
|---|---|---|---|
| DeiT maximum on SoyLocal (results, conclusion, table) | `data/hierarchical_soylocal.csv` | serial 54, `hideit_base_patch16_224.fb_in1k` | maximum |
| DeiT gain over baseline | same | serials 54 and 55, DeiT | maximum 54 − maximum 55 |
| SparK gain over baseline; "up to" gain in the abstract | same | serials 54 and 55, `hiresnet50.in1k_spark`; all backbones for "up to" | maximum 54 − maximum 55; largest over backbones |
| Table, HPH rows, SoyLocal column | same | serial 54, ViT-B (`hivit_base_patch16_224.orig_in21k`) or DeiT | maximum |
| Table, HPH rows, Cotton and SoyAgeing-R1 columns | `data/other_soy_stuff.csv` | serial 86, dataset `cotton` or `soyageing_r1`, ViT-B or DeiT | maximum |
| Table, TransFG, SIM-Trans and CSDNet rows | transcribed from the cited papers | – | not derivable here |
| Gain over CSDNet per dataset | the table | DeiT HPH row minus CSDNet row | difference of printed values |
| Trainable parameters at 10% ratio, per 10% step, 10% to 90% | `data/hierarchical_soylocal.csv` | serial 54, column `no_params_trainable` (millions) | value at ratio 10; (ratio 90 − ratio 10) / 8; ratio 90 − ratio 10 |
| Training images per dataset | `data/hierarchical_soylocal.csv`, `data/other_soy_stuff.csv` | column `num_images_train` | as logged |
| 21 backbones, 11 CNNs, 10 ViTs | `data/hierarchical_soylocal.csv` | serial 54 | distinct `model_name` |

The checker also tests these claims:

- HPH beats the baseline on mean accuracy at every cluster ratio.
- HPH raises every backbone's maximum.
- SparK and DeiT have the largest gains in their architecture groups.
- DeiT has the highest maximum overall.
- SwAV is the only ResNet at or above 60%.
- The SoyAgeing-R1 table values come from the low-resource split.

It compares the three data figures (`compare_soylocal`, `compare_resnet_soylocal`,
`compare_vit_soylocal`) in `drafts/VCIP26/latex/figures/` with the files `plotter.sh` writes
to `results_all/plots/`. A figure passes when at most 1.5% of its pixels differ at the best
alignment within 2 pixels.

This figure check has low specificity:
- Font rendering alone makes up to 1.1% of pixels differ.
- A plot of the same figure from serials 54/55 instead of 39/32, whose boxes shift by about
  1.5 accuracy points, differs by 1.7%.
