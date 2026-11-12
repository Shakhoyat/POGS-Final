# Supervisor report

- `main.tex`: consolidated progress report (status, results of both workstreams, plan).
- `make_figures.py`: rebuilds the new figures and `tables/latency_rows.tex` from cached results
  in this repository (CPU only: numpy + matplotlib) and copies reused figures from
  `report/reproduction/figures/`.

Build from the repository root:

    python report/supervisor/make_figures.py
    cd report/supervisor && tectonic main.tex     # or pdflatex main.tex (run twice), or Overleaf
