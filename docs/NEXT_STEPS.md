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

Two groups of items wait for reruns (next section):
- Items 2, 3, 6 and 12 compare HPH with the serial-55 baselines, which trained with a loss
  bug. Recompute them after `codebase/experiments/rerun_55.sh`.
- Items 7, 9 and 10 use SoyAgeing-R1 values from serial 86. Recompute them after
  `codebase/experiments/rerun_86.sh`.

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
   - the SoyAgeing-R1 cells come from the serial-86 ratio-70 runs. Their pseudo-label files
     were built from all SoyAgeing rounds (4,944 training images), not from R1 (984). The
     ratio-90 runs used the right split: DeiT reaches 86.16 and ViT-B 84.24. DeiT stays best
     in every column.
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

12. **The headline gain, l.177 and l.215.** Decision: restate the gain after the serial-55
    reruns. Reason: 170 of the 210 serial-55 baselines trained on one image per batch (next
    section). The DeiT baseline scores 48.83 with the bug and 63.33 without it (serial 56), so
    most of the 12.5-point gain is the bug. The HPH runs are not affected.
    - Current: "DeiT obtains the largest gain of **12.5%** to max accuracy over the
      non-hierarchical baseline."
    - No replacement text is proposed until `codebase/experiments/rerun_55.sh` has run.

Counts: 12 items. 8 changes are ready to apply (1–7 and 9–10). 4 need a decision: items 8,
11 and 12, and Kevin Tandian's department in item 1. Items 2, 3, 6, 7, 9, 10 and 12 change
again after the reruns.

## Reruns that replace wrong serials (decision)

Three serials are wrong and are replaced in full, under the same serial number. The scripts
in `codebase/experiments/` hold one literal command per run, rebuilt from each run's recorded
launch arguments (`data/run_args/serial_<N>.csv`). Every line parses with the code repo's
argument parser. None has been run.

| Serial | What is wrong | Script | Runs |
|---|---|---|---|
| 55 | SoyLocal baselines trained with a loss bug | `rerun_55.sh` | 210 |
| 86 | SoyAgeing-R1 ratio-70 pseudo-labels built from the wrong split | `rerun_86.sh` | 185 |
| 32 | Configurations of the 21-backbone baseline sweep with no finished run | `rerun_32.sh` | 35 |

**Serial 55, the loss bug.** At code commit `f0c41d8`, `OverallLoss.forward` in
`hierarchical/train_utils/calc_loss.py` computed `self.criterion(output[-1], targets[-1])`.
- A hierarchical model returns a tuple of per-level logits, so `[-1]` selects the finest
  level, as intended.
- A baseline model returns one `[B, C]` tensor, so `[-1]` selects the last image of the batch.
  Each step trained on 1 of 8 images.
- Evaluation was correct, because `compute_hierarchy_accuracy` already wrapped tensors in a
  list.
- Commit `74ef267` wraps tensors in a list before indexing. Checked on CPU: for a `[8, 200]`
  batch, the loss at `f0c41d8` equals the last image's cross-entropy (6.0201), and at
  `74ef267` it equals the batch mean (6.0483).

The flags do not explain the gap. All 140 config keys and the launch arguments of these runs
were compared, DeiT at AdamW 5e-6, 200 epochs, seed 0:

| Serial | Run | Accuracy | Commit |
|---|---|---|---|
| 55, baseline | `fd32sfup` | 48.83 | `f0c41d8` (bug) |
| 56, baseline rerun | `jtetov5g` | 63.33 | `8ad68bc` |
| 54, HPH 90% | `aafc7jba` | 64.33 | `234edfc` |
| 57, HPH 90% reruns | `nmnz973q`, `3xcx9k9w` | 65.50, 61.67 | `8ad68bc`, `d6ed1b7` |

The baselines differ only in `cpu_workers` and `dataset_in_memory`, a flag that `f0c41d8`
lacks and serial 56 leaves off. Of the other baseline serials, none ran before `74ef267`: 32 on
wandb, 56, 59, 61, 81, 83, 85, 94 and 102. Serials 23, 24 and 54 train hierarchical models.

**Serial 86, the wrong split.** `soyageing_r1_hierarchical_pseudo_{deit,vit_b}_70.csv` were
built from `train_val.csv` (all rounds, 4,944 training images) instead of `train_val_R1.csv`
(984). The other datasets and the ratio-90 files are right. Regenerate the two files before
running `rerun_86.sh`.

**Serial 101, the TGDA baseline.** The committed serial-32 rows of 13 backbones were wandb
serial 101: runs of `arkel23/TGDA` `tools/train_teacher.py` on plain timm models, with a
pooled linear head and a square-resize test crop. The hierarchical serial 23 they were paired
with ran on the Hierarchical code. These rows are now replaced by wandb serial 32, which ran
all 21 backbones on the Hierarchical code with serial 23's flags; `rerun_32.sh` fills its
gaps. Old against new, over 584 shared configurations: median difference 0.00, mean +0.53,
393 within 1 point; the per-backbone maximum rises by 0.61 on average.

Procedure, per serial:
1. Move the old wandb runs to `nycu_pcs/Others`. For serial 32, move only its 31 unfinished
   runs, and move serial 101.
2. Run the script from `projects/Hierarchical` at commit `0ce6620` or later.
3. Re-download (the commands are at the end of `plotter.sh`) and compare old and new per
   configuration.
4. Replace the old rows with `codebase.download.replace_serial_rows`; it archives them to
   `/mnt/hdd10tb/edwin/deprecated/`. The serial-101 rows are already archived in
   `/mnt/hdd10tb/edwin/deprecated/hierarchical_buggy_serials_20260929/`.
5. Rerun `plotter.sh`. For 55 and 86, regenerate the VCIP26 figures, rerun
   `codebase.verify.verify_paper_numbers`, and update the VCIP26 list above.
6. Delete the archived rows and the moved wandb runs once the new numbers are settled.

## 4-page resubmission on backbone training strategies (decision)

The goal is a short paper with two claims:
- **(Q1)** how to choose a backbone for a dataset, by top accuracy and by robustness to
  learning rate and optimizer;
- **(Q2)** why hierarchical training is the better choice when hyperparameter robustness
  matters, argued from first principles.

The measurements below come from the committed CSVs. Serial 23 (hierarchical) is paired with
32 (baseline) on aircraft, CUB and cars, and 54 with 55 on SoyLocal: 21 backbones, 84 pairs.
`python -m codebase.analyses.robustness` computes them into `results_all/robustness/`. The
serial-55 SoyLocal baselines trained with the loss bug described above, so the numbers below
use the 63 pairs without SoyLocal (`hier_vs_base_no_soylocal.csv`) until serial 55 is rerun.

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
   - The baselines of 13 backbones came from another codebase (serial 101), and the SoyLocal
     baselines from a buggy loss (serial 55). Both are covered in the reruns section above.
   - `data/hierarchical_pseudo_2.csv` has 53 rows that share a (serial, dataset, model,
     optimizer, learning rate, ratio) key: 4 in serial 32, 49 in serial 40.
   - `summarize_acc` keeps duplicates, so some sweeps average more than 10 runs.
   - ICME26's text contradicts its own table on the IG-1B baseline CV: 6.691 and 7.035 in the
     text, 7.51 and 8.67 in the table.

### What the data already say about Q2

- **Robustness gain.** Over the 63 pairs, hierarchical training lowers mean regret by 2.71
  points under SGD and 1.58 under AdamW, in 67% and 73% of pairs (Wilcoxon p < 0.001 and
  p = 0.001). It raises the mean over the grid by 2.98 and 2.83, in 81% of pairs under each.
- **Accuracy gain.** It raises the maximum by 0.28 under SGD (60% of pairs, p = 0.012) and
  1.25 under AdamW (73%, p = 0.003). The SGD gain is below the 1.05-point seed noise, so the
  paper can claim robustness, and a peak-accuracy gain only under AdamW.
- **Gradient scale is ruled out.** The gain is the same under SGD and AdamW. Adam ignores
  gradient scale, and every run clipped the gradient norm to 1.0. So the summed loss does not
  act as a larger learning rate.
- **Whether the gain grows as data shrinks is open.** SoyLocal (3 images per class) shows the
  largest gain, but against the buggy serial-55 baselines. Serials 102/103 and the serial-55
  reruns test it.

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
   - Kevin Tandian's loss-landscape work on pcs5060ti (see the last section) already
     visualises hierarchical loss surfaces.

Draft motivation for the introduction: "With a few images per class, a mini-batch rarely holds
two images of the same fine class, so the fine-grained loss gives high-variance gradients. A
coarse label pools many images, so its gradient is consistent across batches. Adding it both
regularises the logits toward sibling classes and steadies each capped update, which makes
training less sensitive to the learning rate and optimizer."

### Analyses on existing data (no training)

0. Serials 58 and 59 are the one existing pair that differs only in the hierarchy: aircraft,
   21 backbones, 50 epochs, seed 0, each at its best LR, with 16 of 21 pairs sharing optimizer
   and LR. Hierarchy scores higher in 14 of 21, by +0.45 points on average. It is one LR per
   backbone, so it tests peak accuracy, not robustness.

1. Recompute robustness as regret, basin width and expected maximum, per optimizer and per
   dataset. Report rankings with bootstrap intervals over learning rates.
2. For Q1, use a Pareto plot of maximum against mean regret per dataset as the selection rule.
   Add a variance decomposition: backbone, pretraining objective, optimizer, learning rate and
   dataset, with interactions.
3. Re-test the predictors within each family, against mean regret and maximum, with a
   metadata-only baseline (family and objective) as the bar to beat.
4. Run the Q2 regressions from mechanism 1 (gain against images per class and against cluster
   ratio), and the checkpoint logit-norm and Hessian checks from mechanisms 3 and 4.
5. Deduplicate the CSVs.

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

## Proposed clean-up of the wandb project (decision)

`results_all/serials/serial_triage.csv` gives each of the 86 wandb serials in
`nycu_pcs/Hierarchical` one verdict:
- **KEEP-VCIP:** the serial backs the VCIP26 paper or a fix in the list above.
- **KEEP-EXT:** the extension needs or may need it.
- **KEEP-HISTORY:** an earlier submission used it.
- **MOVE-TO-OTHERS:** a dead end, proposed for `nycu_pcs/Others` to await deletion. Nothing has moved. Moving runs on wandb does not change
the committed `data/*.csv`, so `plotter.sh` still rebuilds everything.

| Verdict | Serials | Runs |
|---|---|---|
| KEEP-VCIP | 13 | 2,618 |
| KEEP-EXT | 37 | 7,082 |
| KEEP-HISTORY | 3 | 647 |
| Ask the owner | 2 | 108 |
| MOVE-TO-OTHERS | 31 | 1,400 |

Proposed moves, 1,400 runs:
- **HiFA heads** (110, 200, 210, 300, 310): 356 runs; their checkpoints are already
  deprecated.
- **Adapters on code never committed** (107, 108): 13 runs.
- **Same-class mix** (91, 93), **an all-failed arm** (95) and **unreferenced SoyLocal runs**
  (99): 133 runs.
- **Pilots and test runs** (29, 104, 120, 121): 14 runs.
- **Feature analyses no paper used** (30, 31, 33–38, 208, 209): 432 runs.
- **Studies replaced by other serials** (41, replaced by 51 for the extractor study; 44 and
  46–49, replaced by 64–69 and 74–79): 452 runs.

Kept, in brief:
- **VCIP26:** 54–57, 81–86, 52/53 and 87. 87 and the newer 52/53 SoyLocal runs answer reviewer
  points on feature quality and on Hungarian matching against other grouping.
- **Extension:**
  - the backbone sweeps (21, 23, 24, 32, 40, 101);
  - the best-learning-rate reruns (58–61);
  - the few-example serials (102, 103);
  - the feature metrics (25–28);
  - the small ablation pairs (42/43, 62–79);
  - the label-smoothing arms (90, 92, 94).
- **History:** 39 (ICME26), 51 and 70 (CVPRW26).
- **Serial 101** was the committed serial-32 baseline of 13 backbones. It moves once
  `rerun_32.sh` has run; the reruns section above has the procedure. The old runs of 55 and
  86 move by the same procedure. The table counts them as kept.

Decisions for the author:
1. Serials 105 and 106 (seed-42 "Metric" runs, 106 still logging on 2026-09-28): ask the owner,
   probably Kevin Tandian.
2. Metric serials 30, 31 and 35–38 (378 runs): move, unless the predictor re-test wants metrics
   on the second target.
3. Serial 41 (401 runs): move, unless the extension revisits the choice of feature extractor.
4. Label-smoothing arms 90, 92 and 94: keep until the planned CE + label-smoothing runs exist,
   then move them with the others (261 runs in all).
5. The 271 failed, crashed or killed runs inside kept serials, listed in
   `results_all/serials/nonfinished_in_keep.csv`: none has a logged accuracy, so they can move
   one by one. Serials 23, 40 and 51 would then show fewer runs than their grids.

## Data and checkpoints

- The checkpoint store's README names serial 84 as the baseline and 85 as the
  pseudo-hierarchy. The checkpoint configs and `data/cotton_seeds.csv` say the opposite.
- The CVPRW26 submission date is not recorded in the repo.
- pcs5060ti holds two checkouts of `github.com/Kev-Tan/Hierarchical_Loss_Landscape`, a
  fork of the code repo, most likely Kevin Tandian's. They are left untouched.
  - `~/Desktop/projects/Hierarchical` has 2 unpushed commits (2026-09-16 and 2026-09-28).
    They add loss-landscape analysis, including `tools/visualize_hierarchical_loss.py`, and
    272 checkpoints, 81 of which match no file on server-3090.
  - `~/Desktop/projects/Hierarchical-clean` has nothing unpushed.
  - The commit listing is `/mnt/hdd10tb/edwin/pcs5060ti_cleanup/hierarchical_unpushed_20260929.txt`.
  - This work is the flatness test for Q2 (mechanism 4 above); coordinate with its owner
    before building another one.
- `results_all/` folders with no recorded command: `diff_tables/`, `soylocal_acc/`,
  `soylocal_hyperparam/`, `val_acc_level1/`, `extracted_text/` and `lr_scripts/`.
- `codebase/analyses/summarize_cost.py` defaults to `--host server-3090`. No serial 23 or 24
  run was logged from that host, so it summarises nothing unless `--host` is given. It is not
  in `plotter.sh`.
- The Hub model cards cite CVGIP 2025 and VCIP 2026 with `eprint = {TBA}`; add the arXiv IDs
  or DOIs when they exist (card builder: `fgir_zoo/tools/build_hierarchical_card.py` in the
  `fgir_zoo` repo).
