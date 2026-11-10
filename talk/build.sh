#!/usr/bin/env bash
# Build the slides of the talk: slides.pdf (with overlay steps).
# The deck reuses the paper's figures (../paper/figures), so run it inside the repository.
set -euo pipefail
cd "$(dirname "$0")"
B=build; mkdir -p $B
for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$B slides.tex >/dev/null; done
cp $B/slides.pdf slides.pdf
if grep -E "^!|Overfull .vbox" $B/slides.log; then echo "WARNING: see $B/slides.log"; fi
echo "slides.pdf: $(pdfinfo slides.pdf | grep Pages)"
# Local extras (not in the repository): handout, speaker script and PowerPoint notes.
if [ -f tools/assemble.py ]; then
  for i in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$B -jobname=slides-handout \
      '\PassOptionsToClass{handout}{beamer}\input{slides}' >/dev/null
  done
  cp $B/slides-handout.pdf slides-handout.pdf
  python3 tools/assemble.py
fi
