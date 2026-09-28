# Next steps

Open items only. Items marked "decision" need the author.

## VCIP26 camera-ready corrections (decision)

`python -m codebase.verify.verify_paper_numbers` fails on the numbers below. Items are in
reading order through `drafts/VCIP26/latex/camera_ready.tex`, with source line numbers. Each
item gives the location, the change and its reason, then the text before and after.

Values follow one rule:
- a printed value is the stored value rounded once, half-up, to the decimals the paper uses
  (one);
- a derived value (a gain) is computed from the printed operands when the paper prints them,
  and otherwise from the stored values, rounded once.

These are proposals; the source is unchanged.

1. **Author block, l.24.** Change: the camera-ready still reads "Anonymous Authors". Kevin
   Tandian is the third author, and Augusto and Edwin share first authorship. Affiliations
   follow the CVGIP25 camera-ready.
   - Before: `\author{Anonymous Authors}`
   - After:
     ```latex
     \author{\IEEEauthorblockN{Augusto Christian Surya\textsuperscript{1,*}, Edwin Arkel Rios\textsuperscript{2,*}, Kevin Tandian\textsuperscript{1}, Bo-Cheng Lai\textsuperscript{2}, Min-Chun Hu\textsuperscript{1}}
     \IEEEauthorblockA{\textsuperscript{1}Department of Computer Science, National Tsing Hua University, Hsinchu, Taiwan\\
     \textsuperscript{2}Department of Electronics Engineering, National Yang Ming Chiao Tung University, Hsinchu, Taiwan\\
     \textsuperscript{*}Equal contribution}}
     ```
   - Kevin Tandian's affiliation is given as the NTHU Department of Computer Science; confirm
     the department. Add e-mails only if VCIP asks for them.
2. **Abstract, l.29.** Change 22.3 to 23.7 and 18.16 to 18.1. Reason: the largest gain is
   SparK, 56.83 − 33.17 = 23.67. The SoyLocal gain over CSDNet from the printed table values
   is 64.3 − 46.2 = 18.1.
   - Before: "achieving absolute gains of up to **22.3%** across backbone. Furthermore, HPH
     consistently improves performance, achieving improvements of up to **18.16%** on
     datasets containing only 3–5 training images per class"
   - After: "achieving absolute gains of up to **23.7%** across backbones. Furthermore, HPH
     consistently improves performance, achieving improvements of up to **18.1%** on datasets
     containing only 3–5 training images per class"
3. **Contributions, l.48.** Change: the same two numbers as item 2, for the same reason.
   - Before: "achieving gains of up to **22.3%** over non-hierarchical baselines and up to
     **18.16%**"
   - After: "achieving gains of up to **23.7%** over non-hierarchical baselines and up to
     **18.1%**"
4. **Setup, l.134.** Change the learning-rate ranges. Reason: the fine-tuning sweeps used
   SGD 0.1–0.003 (4 values) and AdamW 1e-3–5e-6 (6 values). The printed ranges belong to the
   frozen-backbone serial 24, which no VCIP26 result uses.
   - Before: "across 10 learning rates, SGD (0.3–0.003) with weight decay 0, and AdamW
     (0.005–0.000005) with weight decay 0.05."
   - After: "across 10 learning rates, SGD (0.1–0.003, 4 values) with weight decay 0, and
     AdamW (0.001–0.000005, 6 values) with weight decay 0.05."
5. **Parameter overhead, l.170.** Change: name the backbone. Reason: 86.8027M is the DeiT-3
   count; DeiT, the model the results headline, has 86.7851M. The two steps (0.012M per 10%,
   0.096M from 10% to 90%) are correct for either.
   - Before: "the SoyLocal model contains 86.8027M trainable parameters at a cluster ratio of
     10%."
   - After: "the DeiT-3 model contains 86.8027M trainable parameters at a cluster ratio of
     10%."
6. **Results, l.177.** Change three statements. Reasons:
   - three ResNets lose maximum accuracy with HPH: Gluon −2.00, IN21k-MIIL −1.83, MoCo v3
     −1.17; three are unchanged;
   - the SparK gain is 23.7;
   - DeiT-3 IN-1k (+13.3) and ViT MoCo v3 (+12.8) gain more than DeiT (+12.5).
   - Before: "Despite the performance differences between architectures, HPH consistently
     improves the maximum accuracy of CNN and transformer backbones. Among the ResNet models,
     ResNet SparK achieves the largest improvement, increasing the maximum accuracy by
     **22.3%** over the non-hierarchical baseline. For transformer models, DeiT obtains the
     largest gain of **12.5%** to max accuracy over the non-hierarchical baseline."
   - After: "Despite the performance differences between architectures, HPH improves the
     maximum accuracy of 15 of the 21 backbones; three ResNets lose at most 2.0 points. Among
     the ResNet models, ResNet SparK achieves the largest improvement, increasing the maximum
     accuracy by **23.7%** over the non-hierarchical baseline. Among transformers, DeiT-3
     (IN-1k) gains the most, **13.3%**, and DeiT gains **12.5%**."
7. **Table `tab_compare_othersoy`, l.193 and l.196.** Change the ViT-B SoyLocal cell, and
   both HPH SoyAgeing-R1 cells. Reasons:
   - ViT-B's SoyLocal maximum is 58.83;
   - the SoyAgeing-R1 cells come from serial-86 runs trained on the full 4,944-image
     training set. At the 5-per-class split the text describes (984 images), DeiT reaches
     86.16 and ViT-B 84.24. DeiT stays best in every column.
   - Before: "HPH (Ours) & 68.3 & 86.3 & 58.7" and "HPH (Ours) & 72.5 & 89.6 & 64.3"
   - After: "HPH (Ours) & 68.3 & 84.2 & 58.8" and "HPH (Ours) & 72.5 & 86.2 & 64.3"
8. **Dataset sizes, l.206.** Decision: 990 or 984. Reason: the serial-86 low-resource runs
   logged 984 training images; 990 is 198 classes × 5. Keep 990 if it is the dataset's
   official split size.
   - Before: "SoyAgeing-R1 contains 198 classes with 990 training images"
   - After, if the runs' count is used: "SoyAgeing-R1 contains 198 classes with 984 training
     images"
9. **Comparison with CSDNet, l.208.** Change: 18.16 to 18.1, 14.60 to 14.6, and 14.45 to 11.0.
   Reason: one decimal like the table; SoyAgeing-R1 follows item 7 (86.2 − 75.2).
   - Before: "DeiT+HPH improves accuracy by 18.16% on SoyLocal, 14.60% on Cotton, and 14.45%
     on SoyAgeing-R1."
   - After: "DeiT+HPH improves accuracy by 18.1% on SoyLocal, 14.6% on Cotton, and 11.0% on
     SoyAgeing-R1."
10. **Conclusion, l.215.** Change: the same three numbers as item 9, for the same reason.
   - Before: "improving accuracy over the ViT+CSDNet SoTA by **14.60%** on Cotton,
     **14.45%** on SoyAgeing-R1, and **18.16%** on SoyLocal."
   - After: "improving accuracy over the ViT+CSDNet SoTA by **14.6%** on Cotton, **11.0%** on
     SoyAgeing-R1, and **18.1%** on SoyLocal."
11. **Figure `compare_soylocal.png`.** Decision: regenerate from serials 54 and 55. Reason:
    the figure was plotted from the serial 39 (HPH) and 32 (baseline) SoyLocal rows, while the
    text and the other two figures use 54 and 55. The two sets differ slightly: the ratio-10
    lower whisker is 32.2 in the figure and about 33.7 from 54/55. The baseline box is the
    same in both.
    - Before: `plotter.sh` plots `--keep_serials 39 32 --input_file data/hierarchical_pseudo.csv`
    - After: `--keep_serials 55 54 --input_file data/hierarchical_soylocal.csv`, then copy to
      `drafts/VCIP26/latex/figures/`

Counts: 11 items. 8 changes are ready to apply (1–7 and 9–10). 3 need a decision: item 8,
item 11, and Kevin Tandian's department in item 1.

## 4-page resubmission on backbone training strategies (decision)

The goal is a short paper with two claims:
- **(Q1)** how to choose a backbone for a dataset, by top accuracy and by robustness to
  learning rate and optimizer;
- **(Q2)** why hierarchical training is the better choice when hyperparameter robustness
  matters, argued from first principles.

The measurements below come from the committed CSVs. Serial 23 (hierarchical) is paired with
32 (baseline) on aircraft, CUB and cars, and 54 with 55 on SoyLocal: 21 backbones, 84 pairs.
`python -m codebase.analyses.robustness` computes them into `results_all/robustness/`.

### What the earlier analysis got wrong or missed

1. **The robustness metric.** The CV mixes two optimizers with different grids, 6 AdamW
   learning rates against 4 SGD. The "all" column pools the 30 runs of three datasets, so it
   also measures dataset difficulty: ResNet TV1 scores 5.42 pooled against 1.30, 2.39 and
   3.63 per dataset. A CV also rewards a uniformly poor model. Use mean regret to the best run
   of the same backbone (Wortsman et al., ICLR 2024, arXiv 2309.14322), per optimizer and per
   dataset. Add the fraction of runs within 2 points of the best, and the expected maximum of
   k random configurations (Dodge et al., EMNLP 2019).
2. **Ranking by maximum is below the noise floor.**
   - The best and second-best backbones by maximum are 0.03 (CUB), 0.51 (aircraft) and 0.77
     (cars) points apart.
   - The standard deviation over three seeds is about 1.05 points: median over 33
     configurations of serials 84/85, measured on Cotton only.
   - Rankings by maximum disagree across datasets: Kendall τ is −0.08 between aircraft and
     CUB. Rankings by mean over the grid agree: τ 0.67–0.84.
   - So "which backbone is best" by peak accuracy is not identifiable from one seed; by mean or
     regret it is.
3. **The predictors.**
   - The feature-metric search drew 672 regression plots against n = 10–21 backbones, with no
     multiple-comparison correction.
   - For fine-tuned serial 23 it used features of fine-tuned models (`*_ft`) against the same
     runs' accuracy, which is partly circular.
   - All plotted metrics use the train split.
   - Architecture family is a confound: ResNets and ViTs differ in both the metrics and the
     robustness.
   - The literature finds the same instability. Agostinelli et al., ECCV 2022, arXiv
     2204.01403: small setup changes reorder transferability metrics. Singh et al., 2025, arXiv
     2510.06448: metadata heuristics beat them.
4. **The protocol.** One grid serves every backbone at batch size 8, with a 2-epoch warmup,
   no layer-wise learning-rate decay, and gradient-norm clipping at 1.0 in every run. ViTs lose
   most under AdamW: mean regret is 13.7 points under AdamW against 7.0 under SGD in serial 23.
   ResNets lose about 2 points under either optimizer. Known mechanisms fit both findings:
   - ResNets: BatchNorm makes SGD insensitive to the learning rate (Arora, Li, Lyu, ICLR 2019,
     arXiv 1812.03981).
   - ViTs: the patch-embedding gradient causes the SGD/AdamW gap (Kumar et al., ICLR 2024,
     arXiv 2211.09359).
   - ViTs, especially CLIP and MAE: their fine-tuning recipes depend on layer-wise decay (Dong
     et al., arXiv 2212.06138).
5. **Data hygiene.**
   - Serial 32 mixes two run generations from different hosts.
   - `data/hierarchical_pseudo_2.csv` has 53 rows that share a (serial, dataset, model,
     optimizer, learning rate, ratio) key: 4 in serial 32, 49 in serial 40.
   - `summarize_acc` keeps duplicates, so some sweeps average more than 10 runs.
   - ICME26's text contradicts its own table on the IG-1B baseline CV: 6.691 and 7.035 in the
     text, 7.51 and 8.67 in the table.

### What the data already say about Q2

- **Robustness gain.** Hierarchical training lowers mean regret by 4.5 points under SGD and
  4.3 under AdamW, and raises the mean over the grid by 6.4 and 6.1. It is more robust in
  87% (SGD) and 81% (AdamW) of the 84 pairs (Wilcoxon p < 0.001).
- **Accuracy gain.** It also raises the maximum, by 1.9 (SGD) and 1.8 (AdamW), in 79% and 66%
  of pairs. The paper can claim both a robustness and an accuracy gain.
- **Gradient scale is ruled out.** The gain is the same under SGD and AdamW. Adam ignores
  gradient scale, and every run clipped the gradient norm to 1.0. So the summed loss does not
  act as a larger learning rate.
- **The gain grows as data shrinks.** Mean over the grid rises by 12.0 points on SoyLocal (3
  images per class) against 3.0 to 5.0 on CUB, cars and aircraft.

### Candidate mechanisms for Q2, with the test that separates them

1. **Low-data gradient variance (pooling).** At batch size 8 with 200 fine classes, two images
   of the same fine class share a batch about 13% of the time. With 20 coarse groups it is about
   80%. The coarse loss therefore gives a consistent, low-variance gradient that the fine loss
   lacks.
   - Test on existing data: regress the gain on images per class (serials 102/103 hold 1, 3 and
     5) and on the cluster ratio (serials 40 and 54).
2. **Clipped-step sharing.** With the update norm capped, a large, steady coarse gradient takes
   part of every step. This lowers the effective step on the noisy fine loss exactly when the
   learning rate is too high.
   - Test with new runs: clip against no clip, and a summed against a mean loss. The prediction
     is that without clipping, hierarchy at high SGD learning rates loses its advantage.
3. **Structured label smoothing.** The coarse loss raises sibling logits, which bounds the logit
   norm, a known route to divergence at high learning rates.
   - Sources: Müller et al., NeurIPS 2019, arXiv 1906.02629; Wei et al., ICML 2022, arXiv
     2205.09310.
   - Test with new runs: CE with label smoothing, and the code repo's `--ls_hierarchical`, over
     the same grid.
   - Test on existing checkpoints: logit norms of hierarchical against baseline at matched
     learning rates.
4. **Flatness.** A smoother loss tolerates larger steps (Cohen et al., ICLR 2021, arXiv
   2103.00065).
   - Test on existing checkpoints: the top Hessian eigenvalue of the fine loss.

Draft motivation for the introduction: "With a few images per class, a mini-batch rarely holds
two images of the same fine class, so the fine-grained loss gives high-variance gradients. A
coarse label pools many images, so its gradient is consistent across batches. Adding it both
regularises the logits toward sibling classes and steadies each capped update, which makes
training less sensitive to the learning rate and optimizer."

### Analyses on existing data (no training)

1. Recompute robustness as regret, basin width and expected maximum, per optimizer and per
   dataset. Report rankings with bootstrap intervals over learning rates.
2. For Q1, use a Pareto plot of maximum against mean regret per dataset as the selection rule.
   Add a variance decomposition: backbone, pretraining objective, optimizer, learning rate and
   dataset, with interactions.
3. Re-test the predictors within each family, against mean regret and maximum, with a
   metadata-only baseline (family and objective) as the bar to beat.
4. Run the Q2 regressions from mechanism 1 (gain against images per class and against cluster
   ratio), and the checkpoint logit-norm and Hessian checks from mechanisms 3 and 4.
5. Deduplicate the CSVs, and keep one serial-32 generation per comparison.

### New experiments, most convincing first

1. **Mechanism factorial on SoyLocal.** One ResNet-50 and one ViT-B, the full 10-point grid:
   {CE, hierarchical} × {clip 1.0, no clip} × {summed, mean loss}, plus CE with label smoothing.
   Log per-step gradient norms per level, their cosine, and the clipped fraction. One sweep
   separates mechanisms 1–3.
2. **Seeds.** Three seeds at each backbone's best and median configuration, on one dataset,
   for about 6 backbones. This puts intervals on every ranking.
3. **Standard transferability estimators.** Compute LogME (ICML 2021, arXiv 2102.11005), SFDA
   and NCTI (ICCV 2023) on the 21 frozen backbones from forward passes only, and test them
   against maximum and regret.
4. **Protocol fixes on 3–4 ViTs.** Try AdamW with warmup and layer-wise decay, and SGD with the
   patch embedding frozen. Separately, compare frozen against training BatchNorm on 3–4
   ResNets. If ViT fragility disappears, the paper gains a prescription.

## Data and checkpoints

- The checkpoint store's README names serial 84 as the baseline and 85 as the
  pseudo-hierarchy. The checkpoint configs and `data/cotton_seeds.csv` say the opposite.
- The CVPRW26 submission date is not recorded in the repo.
- `results_all/` folders with no recorded command: `diff_tables/`, `soylocal_acc/`,
  `soylocal_hyperparam/`, `val_acc_level1/`, `extracted_text/` and `lr_scripts/`.
- `codebase/analyses/summarize_cost.py` defaults to `--host server-3090`. No serial 23 or 24
  run was logged from that host, so it summarises nothing unless `--host` is given. It is not
  in `plotter.sh`.
- The checkpoint release (repository names, which groups) waits on the FGIR-Backbones Model
  Zoo session's proposal, `/mnt/hdd10tb/edwin/hier_zoo/PROPOSAL.md`.
