# Hierarchical checkpoints: keep / deprecate triage

Scope: the 756 `.pth` in `/mnt/hdd10tb/edwin/results_hierarchical/ckpts/` (the 176 in `deprecated/` are already out).
Nothing was moved. Lists: `keep_list.csv` (222), `deprecate_list.csv` (534). Scripts: `sweep_max.py` → `file_vs_sweep.csv`
(each file against its wandb sweep, final `val_acc_level1`), `build_lists.py` → the two lists.

"At max" = the file's opt+LR is the sweep's best (for reruns, the best of the sweep it reruns). Tail files are kept only
when they are the per-backbone max over LR × ratio of their sweep (the aggregation the papers use); a tail at a
per-ratio max that is not the per-backbone max is deprecated.

| # | Group | Files | GB | Best-LR / fixed-good | Paper(s) | Concrete downstream use | Verdict | Hub repo |
|---|---|---|---|---|---|---|---|---|
| 1 | `real_hierarchy_ckpts`, serial 21, real hierarchy FT, air/cub/cars | 63 | 12.8 | yes, 63/63 at serial-23 max LR, mean −0.13 pt | CVGIP25, ACMMMW25, ICME26 (rerun of their FT sweep) | Only fine-tuned weights of the backbone study. Feature serials 27/28 (CKA, L2 norm, MSC/intra/inter "_ft" columns of `hierarchical_all_{1,2}.csv`, ACMMMW25 fig. wAP vs MSC of fine-tuned models) need a `--ckpt_path`; `cka_metrics/*_27` (63 dirs, "Fine-Tuned") = 21 × 3 exactly, and serial 27 started 2 days after 21 finished. Not recorded in the wandb dump, so "strongly indicated", not proven | KEEP | Hierarchical-Backbones |
| 2 | `serial60_ckpts`, serial 60, real hierarchy FZ | 63 | 12.8 | yes, 60/63 at serial-24 max LR (3 at another LR), mean −1.51 pt; 18 aircraft files at 50 ep | CVGIP25, ACMMMW25, ICME26 (rerun of their FZ sweep) | None for features: backbone checked bit-identical to timm `deit_base_patch16_224.fb_in1k` (149/149 tensors, max diff 0); trained part is the head, 1.13 M of 86.8 M params | KEEP (rule a) | Hierarchical-Backbones |
| 3 | `serial58_ckpts`, aircraft real, 50 ep | 21 | 4.3 | yes, 21/21, mean −0.33 | none | Matched pair with 59 (same backbone, LR, epochs, seed; only hierarchy differs): the one controlled set for a hierarchy-vs-baseline CKA/MSC comparison | KEEP | Hierarchical-Backbones |
| 4 | `serial59_ckpts`, aircraft baseline, 50 ep | 21 | 4.3 | yes, 20/21, mean −0.53 | configuration of CVPRW26's baseline (serial 32) | Only best-LR baseline of the backbone study; pair of 58 | KEEP | Hierarchical-Backbones |
| 5a | `baseline_ckpts`, serial 32 tail, `cars_baseline_hivit_base_patch16_224.dino_32.pth` | 1 | 0.3 | yes (per-backbone max, 88.83) | CVPRW26 baseline | – | KEEP | Hierarchical-Backbones |
| 5b | `baseline_ckpts`, serial 32 tail, rest | 62 | 12.5 | no (AdamW 5e-6 / SGD 0.1 tails; 6 CUB ResNets have no run) | CVPRW26 numbers come from other LRs | – | DEPRECATE | – |
| 6a | `pseudo_hierarchy_ckpts`, serial 40 tail: `aircraft_pl_50_…clip_224.laion2b_40`, `cars_pl_70_…clip_224.laion2b_40` | 2 | 0.6 | yes (per-backbone max over LR × ratio) | CVPRW26 pseudo | – | KEEP | Hierarchical-Backbones |
| 6b | serial 40 tail, rest | 182 | 37.0 | no; 4 more are per-ratio max only | CVPRW26 | – | DEPRECATE | – |
| 7 | `serial61_ckpts`, CUB FZ baseline, 4 backbones | 4 | 0.9 | rerun, but no source sweep on wandb | none | Frozen backbone = public weights; no paper reports an FZ baseline | DEPRECATE | – |
| 8 | `serial102_ckpts`, `serial103_ckpts`, few-example (1-shot aircraft, 2 files 5-shot) | 42 | 8.6 | no, 0/42 at max | none (`model_low_examples.csv` only) | – | DEPRECATE | – |
| 9 | `serial56_ckpts`, SoyLocal baseline best-LR rerun | 21 | 4.3 | yes: 10/21 at the serial-55 max LR; 12/21 score above the 55 max, 9 below (mean +2.83) | configuration of VCIP26's baseline | Pair of 57 (same data, epochs, LR regime): baseline-vs-HPH representation comparison on the paper's main dataset | KEEP | Hungarian-Pseudo-Hierarchies |
| 10 | `serial57_ckpts`, SoyLocal HPH 90% best-LR rerun | 21 | 4.3 | yes: 13/21 at serial-54 ratio-90 max LR, mean −1.25 | configuration of VCIP26's HPH (90%) | Pair of 56 | KEEP | Hungarian-Pseudo-Hierarchies |
| 11a | `serial39_soylocal_ckpts` = wandb 54 tail, per-backbone max over LR × ratio: DeiT r90 (VCIP headline 64.33), DeiT-3 r90, DeiT-3 IN22k r90, DINO r90, MAE r10, CLIP r70 | 6 | 1.9 | yes | VCIP26 (DeiT headline printed; others are the sweep maxima behind its per-backbone comparison) | – | KEEP | Hungarian-Pseudo-Hierarchies |
| 11b | wandb 54 tail, rest (26 more at a per-ratio max, 115 not) | 141 | 28.0 | no | VCIP26 box plots use all 10 LRs, not these files | – | DEPRECATE | – |
| 12a | `baseline_ckpts` SoyLocal = wandb 55 tail at per-backbone max: DeiT-3 IN22k, DINO, CLIP | 3 | 1.0 | yes | VCIP26 baseline maxima; pair of 11a for the same 3 backbones | – | KEEP | Hungarian-Pseudo-Hierarchies |
| 12b | wandb 55 tail, rest | 18 | 3.3 | no | – | – | DEPRECATE | – |
| 13 | `serial81_ckpts`, `serial82_ckpts`, leaf LR search on 80/20 split | 30 | 9.7 | 81: 0/16; 82: 3/14 at max, but on a validation protocol no paper reports | none | – | DEPRECATE | – |
| 14 | `serial83_ckpts`, `serial86_ckpts`, leaf train+val → test, tails | 29 | 9.4 | no, 0/29 (VCIP Cotton/SoyAgeing-R1 maxima are at other LRs) | VCIP26 table numbers, but not these runs | – | DEPRECATE | – |
| 15 | `serial84_ckpts`, `serial85_ckpts`, leaf best LR, seed 100 only | 26 | 8.4 | best LR from the 81/82 search | none (`cotton_seeds.csv`) | – | DEPRECATE (borderline) | – |

## Skeptic check on each KEEP

- **21.** "A best-LR rerun at seed 1, not the paper's run." Fails: the paper's sweep saved no checkpoints, these sit 0.13 pt below it,
  and they are the fine-tuned weights the published FT feature metrics were computed from.
- **60.** "The backbone is the public timm model; these are 200 MB files wrapping a 4.5 MB head." Holds as a
  packaging point, fails as a deprecation argument: they are the best-LR runs of the FZ numbers three papers report (rule a).
  Option for the user: release head-only files.
- **58.** "Redundant with serial 21's aircraft files, which train 200 epochs." Fails: 58 is itself best-LR (−0.33), and 21 differs
  from 59 in epochs and seed, so only 58 gives the controlled hierarchy-vs-baseline pair.
- **59.** "No paper uses serial 59." Fails: it is the best-LR run of the baseline configuration CVPRW26 reports; the serial-32
  tails are not.
- **56.** "11/21 are not at the serial-55 max LR." Fails: 12/21 score above the 55 max (mean +2.83), so as a set they are the best
  baseline models that exist.
- **57.** "8/21 are not at the 54 max LR." Fails: mean −1.25, and 57 is the paper's chosen ratio at a best-LR setting for all 21.
- **11a / 12a / 5a / 6a (11 tail files).** "Sweep tails." Fails: each is the run that the paper's max-over-LR aggregation selects.
  For DeiT-3 IN1k r90 and DINO r90, serial 57 holds the same LR (61.50 / 60.00 vs 61.83 / 61.00); they are kept as the sweep's own max.

## Totals

| | Files | GB |
|---|---|---|
| KEEP → ERISLab/Hierarchical-Backbones (21, 60, 58, 59, 1 of 32, 2 of 40) | 171 | 35.2 |
| KEEP → ERISLab/Hungarian-Pseudo-Hierarchies (56, 57, 6 of 54, 3 of 55) | 51 | 11.5 |
| DEPRECATE | 534 | 117.9 |
| Total | 756 | 164.6 |

## Caveats

- The sweep max is taken from the current wandb runs (`wandb_runs.csv`), not the analysis CSVs. For serial 32 the CSVs
  mix two run generations, so "at max" for 32 is against the current generation.
- Serial 56 DeiT scores 63.33 at AdamW 5e-6, where the serial-55 run of the same configuration scored 48.83 and VCIP26's
  printed baseline is 51.83 (SGD 0.01). The VCIP headline gain (64.33 vs 51.83) is within this run-to-run spread;
  serial 57 DeiT at ratio 90 scores 65.50.
