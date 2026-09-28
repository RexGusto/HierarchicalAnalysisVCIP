# Next steps

Open items only. Items marked "decision" need the author.

## VCIP26 camera-ready: numbers and claims the data do not support (decision)

`python -m codebase.verify.verify_paper_numbers` fails on these items. Each gives the text
as printed, then the value the data give.

1. "HPH achieves an absolute accuracy improvement of up to 22.3%" (abstract, contributions,
   results) and "ResNet SparK ... increasing the maximum accuracy by 22.3%". The SparK maximum
   is 56.83 with HPH (serial 54) and 33.17 without (serial 55), a gain of 23.7 (23.6 from the
   rounded operands).
2. Table "HPH (Ours)", ViT-B row, SoyLocal column: "58.7". The ViT-B (IN-21k) maximum on
   serial 54 is 58.83, which prints as 58.8.
3. "HPH consistently improves the maximum accuracy of CNN and transformer backbones." Three
   ResNets lose maximum accuracy with HPH: Gluon −2.00, IN21k-MIIL −1.83 and MoCo v3 −1.17.
   Three are unchanged: DeiT-3 IN-22k, TV2 and MAE. The mean accuracy over each backbone's
   sweep does rise at every cluster ratio.
4. "For transformer models, DeiT obtains the largest gain of 12.5%." Two transformers gain
   more: DeiT-3 IN-1k +13.33 and ViT MoCo v3 +12.83.
5. Table SoyAgeing-R1 column, "89.6" (DeiT) and "86.3" (ViT). Both come from serial-86 rows
   trained on 4,944 images. The rows at the low-resource split, 984 images, which the text
   describes as 5 per class, peak at 86.16 (DeiT) and 84.24 (ViT). The text also gives 990
   training images, while the serial-86 low-resource rows record 984.
6. "improves accuracy by 18.16% on SoyLocal ... 14.45% on SoyAgeing-R1". From the printed
   operands these are 64.3 − 46.2 = 18.1 and 89.6 − 75.2 = 14.4. The paper computed them from
   unrounded values and prints two decimals against one-decimal operands.
7. "86.8027M trainable parameters at a cluster ratio of 10%" is the DeiT-3 count. The DeiT-B
   model the results section headlines has 86.7851M. The sentence does not name the backbone.
8. The camera-ready source still reads "Anonymous Authors".

## VCIP26 figures

- `drafts/VCIP26/latex/figures/compare_soylocal.png` (the cluster-ratio figure) is plotted
  from the serial 39 (HPH) and 32 (baseline) SoyLocal rows in `data/hierarchical_pseudo.csv`.
  The results text and the other two figures use serials 54 and 55. The serials deck says the
  SoyLocal runs moved from 39 and 32 to 54 and 55, but the two sets are not identical. Plotted
  from 54 and 55, the HPH boxes at ratios 10, 50 and 70 shift slightly: the ratio-10 lower
  whisker is about 33.7 instead of 32.2. Decide whether the figure should be regenerated from
  54 and 55. The baseline box is the same in both.

## Data and checkpoints

- The learning-rate sweeps saved only their last run, AdamW 5e-6. A max-over-learning-rate
  number therefore has a checkpoint only when its best run was that last one. Coverage by
  serial:
  - **Serials 23 and 24** (the backbone results of CVGIP25, ACMMMW25 and ICME26) have no
    checkpoints on this box, ed-storage or ed520.
  - **Serial 54 (SoyLocal HPH).** Its last runs are the 147 files in
    `results_hierarchical/ckpts/serial39_soylocal_ckpts`, which still carry serial 39 inside
    the file; 140 match their serial-54 rows exactly. Six of the 21 backbones reach their
    maximum at that learning rate: DeiT (the 64.33 headline, ratio 90), both DeiT-3 models,
    DINO, MAE and CLIP.
  - **Serial 55 (SoyLocal baseline).** Its last runs are the 21 SoyLocal files in
    `ckpts/baseline_ckpts`. They carry serial 32 inside the file, and all 21 equal their
    serial-55 rows.
- The checkpoint store's README names serial 84 as the baseline and 85 as the
  pseudo-hierarchy. The checkpoint configs and `data/cotton_seeds.csv` say the opposite.
- The CVGIP25 outcome and the CVPRW26 submission date are not recorded in the repo.
- `results_all/` folders with no recorded command: `diff_tables/`, `soylocal_acc/`,
  `soylocal_hyperparam/`, `val_acc_level1/`, `extracted_text/` and `lr_scripts/`.
- `codebase/analyses/summarize_cost.py` defaults to `--host server-3090`. No serial 23 or 24
  run was logged from that host, so it summarises nothing unless `--host` is given. It is not
  in `plotter.sh`.
