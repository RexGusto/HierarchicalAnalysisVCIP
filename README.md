# HierarchicalAnalysis

This repo holds the analysis code for five papers on hierarchical fine-grained image
recognition (FGIR). In hierarchical FGIR, a classifier predicts a coarse label, such as the
aircraft manufacturer, together with the fine label, such as the model.

The papers ask two questions:
- How do the backbone, its pretraining and the optimisation hyperparameters change accuracy
  and robustness?
- Can a pseudo-hierarchy, built automatically by grouping similar classes, replace an
  annotated hierarchy?

The final paper, VCIP 2026, proposes Hungarian Pseudo-Hierarchies (HPH) for
ultra-fine-grained leaf datasets. The paper lineage and the reviews are in
`docs/SUBMISSION_HISTORY.md`.

The repo does not train models. Training ran in the code repo `projects/Hierarchical`. Each
run was logged to the Weights & Biases project `nycu_pcs/Hierarchical`, and the runs were
exported to the CSVs in `data/`. This repo turns those CSVs into the summary tables in
`results_all/`, the figures, and the checks on the VCIP 2026 paper.

## Running it

From the repo root, in the `asr` env:

```bash
bash plotter.sh                      # every stage
bash plotter.sh vcip paper verify    # one or more stages
```

The stages are:
- `backbones`: CVGIP25, ACMMMW25 and ICME26;
- `pseudo`: ICME26 and CVPRW26;
- `vcip`;
- `few_examples`;
- `robustness`: hierarchical against baseline per optimizer, ranking stability and seed noise;
- `paper`: builds the VCIP26 PDF;
- `verify`: re-derives the paper's numbers.

The `backbones` stage draws about 730 figures and takes the longest. The wandb downloads
that produced `data/` are listed, commented out, at the end of `plotter.sh`. They need an
API key.

Other checks:

```bash
python -m codebase.verify.verify_paper_numbers     # VCIP26 numbers, claims and figures
python -m codebase.verify.diff_results             # regenerated CSVs vs HEAD, by value
python -m codebase.verify.verify_checkpoints --ckpt_root /mnt/hdd10tb/edwin/results_hierarchical \
    --code_repo /mnt/ssd2tb/edwin/projects/Hierarchical   # fgir_timm912 env
```

## Layout

| Path | Holds |
|---|---|
| `plotter.sh` | The pipeline: every summary table and figure from `data/`, then the VCIP26 build and checks |
| `utils.py` | Shared name dictionaries, filtering and sorting used by every script |
| `claude_process.sh` | One-off commands, kept commented out once run |
| `codebase/` | All other code (see below) |
| `data/` | wandb exports, one CSV per group of serials |
| `results_all/` | Everything the pipeline writes: `acc/` summaries, `pseudo_hyperparam/` differences, `plots/`, `cvgip/`, `new_plots/`, `checkpoints/` |
| `drafts/` | One folder per submission with its PDF and LaTeX source; `drafts/README.md` is the lineage table |
| `reviews/` | Reviews per submission as markdown; `reviews/originals/` keeps the slide decks they came from |
| `docs/` | `SERIALS.md`, `SUBMISSION_HISTORY.md`, `FINDINGS.md`, `NUMBERS.md`, `CHECKPOINTS.md`, `NEXT_STEPS.md`, `CLAUDE_CHANGES.md`; `docs/sources/` keeps the serials slide deck and its markdown conversion |
| `deprecated/` | Retired code, data and duplicate PDFs, listed in `deprecated/README.md` |

## Code

| Module | Does |
|---|---|
| `codebase/download/download_save_wandb_data.py` | Exports runs of given serials to `data/<name>.csv` |
| `codebase/download/download_save_wandb_data_feature_metrics.py` | Exports feature-metric runs (CKA, distances) |
| `codebase/download/dump_wandb_runs.py` | Indexes every run of the wandb project into `data/wandb_runs.csv` |
| `codebase/download/merge_acc_metrics_stats.py` | Joins accuracy, feature metrics and pretraining statistics into `data/hierarchical_all_*.csv` |
| `codebase/download/update_wandb_project_serialx_to_serialy_based_on_substr_in_field.py` | Moves runs between serials on wandb |
| `codebase/analyses/summarize_acc.py` | Per serial, dataset and backbone: mean, max, standard deviation and ADA ratio over the learning-rate sweep |
| `codebase/analyses/summarize_acc_metrics.py` | The same, joined with feature metrics |
| `codebase/analyses/compute_diff.py` | Pseudo-hierarchy minus baseline tables |
| `codebase/analyses/robustness.py` | Hierarchical against baseline per optimizer (maximum, mean, regret, basin width, collapse rate), ranking stability across datasets, seed noise |
| `codebase/analyses/summarize_cost.py`, `get_corr_csv.py`, `acc_deviation_average_ratio.py`, `compute_correlations.py` | Cost summaries, correlation tables, ADA-ratio analysis, regression helper |
| `codebase/plots/plot.py` | Every figure: box, bar, line, regression and heatmap |
| `codebase/plots/merge_corr_plots.py` | Tiles regression plots into grids |
| `codebase/build/lr_script.py` | Writes training command files for learning-rate sweeps |
| `codebase/build/pptx_to_md.py` | Converts a slide deck to markdown with its images |
| `codebase/build/extract_table_from_pdf.py` | Extracts text tables from a PDF page |
| `codebase/verify/*` | The three checks above, and `reconcile_serials.py`, which counts every wandb serial against `data/`, `docs/SERIALS.md` and the checkpoints |
| `codebase/logs/*.sh` | The command logs `plotter.sh` was built from |

The ADA ratio is 100 × standard deviation / mean accuracy over a learning-rate sweep. It is
the coefficient of variation, which is the name the later papers use.

## Known gaps

The VCIP26 camera-ready prints several numbers and claims that the data do not support. They
are listed in `docs/NEXT_STEPS.md` and `docs/FINDINGS.md`. Some `results_all/` folders have
no recorded command: `diff_tables/`, `soylocal_acc/`, `soylocal_hyperparam/`,
`val_acc_level1/`, `extracted_text/` and `lr_scripts/`. They are kept as committed.
