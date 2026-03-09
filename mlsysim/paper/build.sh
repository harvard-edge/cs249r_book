#!/usr/bin/env bash
# Build mlsysim paper: pdflatex → bibtex → pdflatex × 2
# Usage: ./build.sh [clean]
set -uo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "clean" ]]; then
    rm -f paper.{aux,bbl,blg,log,out,pdf,toc,lof,lot}
    echo "Cleaned build artifacts."
    exit 0
fi

echo "Pass 1/4: pdflatex..."
pdflatex -interaction=nonstopmode paper.tex > /dev/null 2>&1 || true

echo "Pass 2/4: bibtex..."
bibtex paper > /dev/null 2>&1 || true

echo "Pass 3/4: pdflatex..."
pdflatex -interaction=nonstopmode paper.tex > /dev/null 2>&1 || true

echo "Pass 4/4: pdflatex..."
pdflatex -interaction=nonstopmode paper.tex > /dev/null 2>&1 || true

if [[ ! -f paper.pdf ]]; then
    echo "ERROR: paper.pdf not generated. Check paper.log for details."
    exit 1
fi

PAGES=$(pdfinfo paper.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "?")
SIZE=$(du -h paper.pdf | awk '{print $1}')
echo "Done: paper.pdf (${PAGES} pages, ${SIZE})"
