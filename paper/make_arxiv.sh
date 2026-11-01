#!/usr/bin/env bash
# Build an arXiv source package (arxiv.tar.gz) for the paper and test-compile it in a clean directory.
# arXiv does not run BibTeX, so the package ships the generated main.bbl.
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -quiet main.tex >/dev/null
OUT="${1:-$PWD/arxiv.tar.gz}"
TMP="$(mktemp -d)"
mkdir -p "$TMP/src/figures"
cp main.tex main.bbl refs.bib "$TMP/src/"
cp figures/*.tex "$TMP/src/figures/"
tar -czf "$OUT" -C "$TMP/src" main.tex main.bbl refs.bib figures
mkdir "$TMP/test" && tar -xzf "$OUT" -C "$TMP/test"
( cd "$TMP/test" && for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex >/dev/null; done )
if grep -E "^!|undefined|Rerun to get" "$TMP/test/main.log"; then echo "test compile FAILED"; exit 1; fi
echo "package: $OUT ($(tar -tzf "$OUT" | wc -l) files), test compile OK, $(pdfinfo "$TMP/test/main.pdf" | grep Pages)"
rm -rf "$TMP"
