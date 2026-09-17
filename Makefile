# Pipeline entry points. Run from the repository root.
#
# Targets are independent so any stage can be re-run on its own, which means make will
# not chase prerequisites for you: the order in `all` is the order they have to run in.
# See scripts/README.md for what each stage costs.

PY := paleo/bin/python

.PHONY: all test smoke check 3dvar hgaoenkf evidence mt ablations product figures \
        trajectories

all: check 3dvar hgaoenkf evidence mt ablations product figures

test:
	$(PY) -m pytest -q

# Every stage at reduced size, into outputs/_smoke, in a few minutes. Exercises the
# wiring and the figure code before committing to a full run.
smoke:
	$(PY) scripts/00_check_data.py
	$(PY) scripts/01_3dvar.py --smoke
	$(PY) scripts/02_hgaoenkf.py --smoke
	$(PY) scripts/03_hgaoenkf_evidence.py --smoke
	$(PY) scripts/04_hgaoenkf_mt.py --smoke
	$(PY) scripts/05_ablations.py --smoke
	$(PY) scripts/06_product.py --smoke
	$(PY) scripts/07_figures.py --smoke

# Re-run one lane against the operating points already on disk. Every script takes
# `--only <stage>`; `make <script>` with no flag runs all of its stages.
trajectories:
	$(PY) scripts/01_3dvar.py --only trajectory
	$(PY) scripts/02_hgaoenkf.py --only trajectory
	$(PY) scripts/03_hgaoenkf_evidence.py --only trajectory
	$(PY) scripts/04_hgaoenkf_mt.py --only trajectory
	$(PY) scripts/07_figures.py

check:      ; $(PY) scripts/00_check_data.py
3dvar:      ; $(PY) scripts/01_3dvar.py
hgaoenkf:   ; $(PY) scripts/02_hgaoenkf.py
evidence:   ; $(PY) scripts/03_hgaoenkf_evidence.py
mt:         ; $(PY) scripts/04_hgaoenkf_mt.py
ablations:  ; $(PY) scripts/05_ablations.py
product:    ; $(PY) scripts/06_product.py
figures:    ; $(PY) scripts/07_figures.py
