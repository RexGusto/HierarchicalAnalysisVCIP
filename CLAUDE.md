# Hierarchical_plot — conventions

Analysis repo for the hierarchical fine-grained recognition papers (CVGIP25, ACMMMW25,
ICME26, CVPRW26, VCIP26). See `README.md` for the layout and how to run it. Inherits the
shared conventions in `/mnt/ssd2tb/edwin/analysis/CLAUDE.md` and the box rules in
`~/.claude/CLAUDE.md`; where they disagree, the parent files win. The paper and document
rules are in `/mnt/ssd2tb/edwin/analysis/WRITING_RULES.md`.

## Environment

- Analysis, plots and the paper checker run in `asr` (the only env with seaborn, statsmodels
  and pypdf). `codebase.verify.verify_checkpoints` runs in `fgir_timm912`, because it builds
  models with the code repo's timm 0.9.12 classes.
- `plotter.sh` exports `MPLBACKEND=Agg`. Sessions here inherit a forwarded X display, and
  without Agg every figure round-trips over it (minutes per plot instead of seconds).
- The VCIP26 paper builds with `bash drafts/VCIP26/build.sh` (pdflatex, TeX Live 2026 on the
  HDD). The other submissions' sources in `drafts/*/latex/` are frozen at their submitted
  state; they build with the same pdflatex/bibtex sequence.

## Invocation: scripts are modules, run from the repo root

`python -m codebase.<sub>.<mod>`, always from the repo root; every data path is
root-relative. `utils.py` stays at the root as the shared library. `codebase/` is grouped by
what a script produces: `download/` fills `data/` from wandb, `analyses/` writes a CSV,
`plots/` writes a PNG, `build/` writes a `.sh` or `.md`, `verify/` writes only a report and an
exit code. `codebase/logs/` holds the three command logs the pipeline was reconstructed from;
they are history, not entry points, and their paths predate the move into `codebase/`.

## Data and serials

- `data/*.csv` are wandb exports of the project `nycu_pcs/Hierarchical`, one or more serials
  per file; `docs/SERIALS.md` maps serials to experiments, CSVs and papers. Several CSVs
  overlap (serial 23 is in eight of them); the file a plot reads is the one named in
  `plotter.sh`.
- `utils.SERIAL_DIC` sets legend labels per serial. Serials 54/55 must stay
  "Pseudo-Hierarchy"/"Baseline" for the VCIP26 figures to reproduce.
- `utils.filter_df` fills a missing `n_cluster_ratio` with 0 so baselines plot at ratio 0;
  `--keep_ratios 0 ...` and the VCIP26 `compare_soylocal` figure depend on it.

## Checks

- `codebase.verify.verify_paper_numbers` re-derives every data-backed number and claim in
  `drafts/VCIP26/latex/camera_ready.tex` and compares the three data figures to the submitted
  ones by pixel; it exits nonzero on any mismatch. Known mismatches are listed in
  `docs/NEXT_STEPS.md`; do not edit the camera-ready text to make the checker pass without
  the author's decision.
- `codebase.verify.diff_results` compares regenerated CSVs under `results_all/` to `HEAD` by
  value, so a rerun that only reorders rows or adds columns is recognised as unchanged.
- `codebase.verify.verify_checkpoints` matches the checkpoints in
  `/mnt/hdd10tb/edwin/results_hierarchical` to their CSV rows and strict-loads a sample; its
  output is `results_all/checkpoints/`.

## The code repo behind this analysis

Training code is `/mnt/ssd2tb/edwin/projects/Hierarchical` (models `hiresnet50.*`,
`hivit_*`, `hideit*`; the head returns one logit tensor per level). Ask before modifying it.
Checkpoint publication is handled by the FGIR-Backbones Model Zoo session; this repo only
reads the checkpoints.
