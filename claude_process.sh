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
