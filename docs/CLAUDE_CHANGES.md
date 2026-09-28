# Change log

Dated record of substantive changes. Open items are in `NEXT_STEPS.md`.

## 2026-09-28 — restructure into the standard analysis-repo layout

- **Layout.** The repo now uses the `ILA_SAW_Analysis` / `BackbonesAnalysis` layout.
  - Code moved from the root into `codebase/{download,analyses,plots,build,verify}` and runs
    as `python -m` from the root.
  - The three command logs (`auto_process.sh`, `cvgip.sh`, `pseudo_labels.sh`) moved
    verbatim to `codebase/logs/`.
  - `paper/`, `submissions/` and `drafts/main_v{1,2}.pdf` became `drafts/<VENUE>/`.
  - `old/` became `deprecated/old/`, and `compute_z_score.py` was retired (see
    `deprecated/README.md`).
- **Submission sources.** The LaTeX sources of CVGIP25, ACMMMW25, ICME26 and CVPRW26 were
  extracted from the zips the author supplied. The CVPRW26 PDF and submission were added to
  the lineage.
  - Every source builds with pdflatex. ICME26, CVPRW26, ACMMMW25 and both VCIP26 versions
    reproduce their submitted PDFs' text; the remaining differences are layout only.
  - The CVGIP25 source is a later revision of the submitted PDF.
  - TeX Live gained `algorithm2e`, `ifoddpage`, `overpic`, `paralist`, `relsize`, `eepic` and
    `latexdiff`.
- **Reviews and decks.** The ICME26 and VCIP26 review decks and the serials deck were
  converted to markdown with `codebase/build/pptx_to_md.py`; the originals were kept. The
  ACMMM-AOCV OpenReview paste became `reviews/review_hierarchical_acmmmw25.md`.
- **Driver.** `plotter.sh` was reconstructed from the three logs, the commented commands
  included.
- **Bugs fixed in `utils.py` and `summarize_acc.py`.**
  - The few-examples commit (`e13cb1f`) had listed `seed` and `n_cluster_ratio` twice in
    `keep_columns`. The duplicate columns broke every heatmap and every plot with either on an
    axis.
  - The same commit disabled the fill of a missing `n_cluster_ratio` with 0, which dropped
    baselines from ratio plots.
  - It also relabelled serials 54/55 as BL/PH, so the VCIP26 figures no longer reproduced.
    All three were restored.
  - `summarize_acc` assumed pseudo-hierarchy columns, so it failed on the serial 23/24 CSVs.
    It now checks for them.
- **Stale defaults.** The defaults in `plot.py`, `get_corr_csv.py` and
  `extract_table_from_pdf.py` pointed at files that exist only under `deprecated/old/`; they
  now point at current files.
- **Environment.** `statsmodels==0.14.5` (and `patsy`) was installed into `asr`. Before
  that, no env on the box could import `plot.py`.
- **Checks added.**
  - `verify_paper_numbers`: re-derives the VCIP26 numbers, claims and figures.
  - `diff_results`: compares regenerated CSVs to `HEAD` by value.
  - `verify_checkpoints`: matches the 932 checkpoints in
    `/mnt/hdd10tb/edwin/results_hierarchical` to CSV rows and strict-loads two per folder.
- **Docs.** `README.md`, `CLAUDE.md`, `requirements.txt` and every file in `docs/` were
  written from the decks, the CSVs and the paper sources.
