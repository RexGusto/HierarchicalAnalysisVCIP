#!/bin/bash
# Builds latex/camera_ready.pdf (pdflatex, bibtex, pdflatex x2) and checks for errors and undefined references.
# Run from anywhere: bash drafts/VCIP26/build.sh
set -euo pipefail
cd "$(dirname "$0")/latex"
export PATH=/mnt/hdd10tb/edwin/texlive/2026/bin/x86_64-linux:$PATH
pdflatex -interaction=nonstopmode camera_ready.tex > build.log
bibtex camera_ready >> build.log
pdflatex -interaction=nonstopmode camera_ready.tex >> build.log
pdflatex -interaction=nonstopmode camera_ready.tex >> build.log
! grep -n '^!' build.log
! grep -n 'undefined' camera_ready.log
pdfinfo camera_ready.pdf | grep Pages
