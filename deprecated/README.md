# deprecated

Retired material, kept for reference. Nothing in `codebase/` or `plotter.sh` reads from here,
except that `codebase/build/extract_table_from_pdf.py` defaults to `old/papers/which_tokens.pdf`.

| Path | What it is | Retired | Why | To restore |
|---|---|---|---|---|
| `old/` | Early scripts (`download_wandb_merge.py`, `summarize_acc_og.py`), early CSVs, the first `results_all/` tree (194 MB), and two reference papers (`papers/`) | 2026-09-28 | Superseded by `codebase/` and the current `results_all/`; was the top-level `old/` | `git mv deprecated/old old` |
| `compute_z_score.py` | z-scores of the accuracy summaries | 2026-09-28 | Imports `DATASETS_UFGIR`, which `utils.py` no longer defines; no driver calls it; its outputs exist only in `old/results_all/z_score` | Add `DATASETS_UFGIR` to `utils.py`, move to `codebase/analyses/` |
| `duplicate_drafts/main_v1.pdf`, `main_v2.pdf` | Byte-identical copies of `drafts/VCIP26/main_v1_submission.pdf` and `main_v2_camera_ready.pdf` | 2026-09-28 | Duplicates (checked with `cmp`) | Not needed |
| `submissions_README.md` | One-line paper progression | 2026-09-28 | Replaced by `drafts/README.md` and `docs/SUBMISSION_HISTORY.md` | Not needed |
