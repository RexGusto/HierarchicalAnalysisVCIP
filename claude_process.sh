#!/bin/bash
# One-off commands, newest last; spent ones stay commented out. Dry-run by default.
# Run from the repo root: bash claude_process.sh [--execute]
set -euo pipefail
PY="${PY:-python}"
run() { if [ "${1:-}" = "--execute" ]; then shift; "$@"; else shift; echo "[dry-run] $*"; fi; }
X="${1:-}"

# 2026-09-28: convert the review and serials slide decks to markdown (originals kept in reviews/originals/ and docs/sources/)
# run "$X" $PY -m codebase.build.pptx_to_md --input_file "reviews/originals/ICME26 Paper Reviews - 23 Feb 2026.pptx" --output_file reviews/review_hierarchical_icme26.md --title "ICME 2026 reviews: Pseudo-Hierarchies for Robust Backbone Selection in Fine-Grained Image Recognition"
# run "$X" $PY -m codebase.build.pptx_to_md --input_file "reviews/originals/VCIP paper review.pptx" --output_file reviews/review_hierarchical_vcip26.md --title "VCIP 2026 reviews: Hungarian Pseudo-Hierarchies for Coarse-to-Fine Ultra Fine-Grained Image Recognition"
# run "$X" $PY -m codebase.build.pptx_to_md --input_file "docs/sources/Pseudo-Hierarchy Serials - 28 Apr 2026.pptx" --output_file docs/sources/serials_deck.md --title "Pseudo-hierarchy project progress and serials (slide deck)"

# 2026-09-29: replace the serial-101 rows (TGDA codebase) stored as serial 32 with wandb serial 32 (Hierarchical codebase); old rows archived on the HDD
# run "$X" $PY -m codebase.download.replace_serial_rows --serial 32 --fresh_file data/hierarchical_serial32.csv --targets data/hierarchical_real.csv data/hierarchical_pseudo_2.csv data/hierarchical_pseudo.csv data/hierarchical_baseline.csv --datasets aircraft cars cub aircraft_baseline cars_baseline cub_baseline --archive_dir /mnt/hdd10tb/edwin/deprecated/hierarchical_buggy_serials_20260929/data --execute
# 2026-09-29: rerun scripts for the serials to replace (codebase/experiments/)
# run "$X" $PY -m codebase.download.dump_run_args --serials 55 86 32 101
# run "$X" $PY -m codebase.build.rerun_scripts --serial 55 --run_args data/run_args/serial_55.csv --commit 0ce6620 --note "The original runs trained with the loss bug fixed in 74ef267 (loss on the last image of each batch only)."
# run "$X" $PY -m codebase.build.rerun_scripts --serial 86 --run_args data/run_args/serial_86.csv --commit 0ce6620 --note "First regenerate soyageing_r1_hierarchical_pseudo_{deit,vit_b}_70.csv from train_val_R1.csv (990 images); the old files were built from train_val.csv (all rounds, 4,950 images)."
# run "$X" $PY -m codebase.build.rerun_scripts --serial 32 --run_args data/run_args/serial_32.csv --gaps_only_from 101 --commit 0ce6620 --note "Only configurations with no finished serial-32 run; serial 101 (TGDA-codebase baseline) is replaced by serial 32."
