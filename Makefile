# Pipeline entry points. Run from the repository root.
#
# Targets are independent so any stage can be re-run on its own, which means make will
# not chase prerequisites for you: the order in `all` is the order they have to run in.
# See scripts/README.md for what each stage inherits.

PY := paleo/bin/python

.PHONY: all test smoke check 3dvar evidence mta product figures trajectories

all: check 3dvar evidence mta product figures

test:
	$(PY) -m pytest

# Every stage at reduced size, into outputs/_smoke, then the results notebook against those
# artefacts. Exercises the wiring before committing to a full run. The executed notebook is
# written to outputs/_smoke, so the tracked copy is left as it is.
SMOKE_OUT := outputs/_smoke
smoke:
	$(PY) scripts/00_check_data.py
	$(PY) scripts/01_3dvar.py --smoke
	$(PY) scripts/02_evidence.py --smoke
	$(PY) scripts/03_mta.py --smoke
	$(PY) scripts/04_product.py --smoke
	PALEORECO_OUTPUTS=$(SMOKE_OUT) $(THREADS) $(PY) -m nbconvert --to notebook --execute \
	    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 \
	    --output-dir $(SMOKE_OUT) notebooks/final_results.ipynb

# Re-run the time-series lane against the operating points already on disk. Every script
# takes `--only <stage>`; `make <script>` with no flag runs all of its stages.
trajectories:
	$(PY) scripts/01_3dvar.py --only time-series
	$(PY) scripts/02_evidence.py --only time-series
	$(PY) scripts/03_mta.py --only time-series
	$(MAKE) figures

check:      ; $(PY) scripts/00_check_data.py
3dvar:      ; $(PY) scripts/01_3dvar.py
evidence:   ; $(PY) scripts/02_evidence.py
mta:        ; $(PY) scripts/03_mta.py
product:    ; $(PY) scripts/04_product.py

# Executes the results notebook in place: every report figure, table and quoted number,
# read from the stored artefacts. Single-threaded BLAS keeps the numbers identical run to run.
THREADS := OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
           NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
figures:
	$(THREADS) $(PY) -m nbconvert --to notebook --execute --inplace \
	    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 \
	    --ExecutePreprocessor.record_timing=False notebooks/final_results.ipynb
