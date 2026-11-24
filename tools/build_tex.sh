#!/usr/bin/env bash
# Compile the LaTeX documents with the user-level TinyTeX (~/.TinyTeX). Usage: tools/build_tex.sh [report|paper|all]
set -u
export PATH="$(echo ~/.TinyTeX/bin/*):$PATH"
cd "$(dirname "$0")/.."
build() {  # dir root
  ( cd "$1" && pdflatex -interaction=nonstopmode -halt-on-error "$2.tex" >/dev/null \
    && { [ -f refs.bib ] && bibtex "$2" >/dev/null || true; } \
    && pdflatex -interaction=nonstopmode -halt-on-error "$2.tex" >/dev/null \
    && pdflatex -interaction=nonstopmode -halt-on-error "$2.tex" >/dev/null ) \
    && echo "ok   $1/$2.pdf  ($(grep -c 'Overfull' "$1/$2.log") overfull, $(grep -c 'undefined' "$1/$2.log") undefined)" \
    || { echo "FAIL $1/$2"; grep -A3 '^!' "$1/$2.log" | head -20; return 1; }
}
case "${1:-all}" in
  report) build docs/report progress_report ;;
  paper)  build docs/paper_draft main ;;
  all)    build docs/report progress_report; build docs/paper_draft main ;;
esac
