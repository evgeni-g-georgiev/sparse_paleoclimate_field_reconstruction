# Pipeline

Sparse paleoclimate field reconstruction by data assimilation: three evaluation lanes
over four estimators, the reconstruction they select, and the report figures.

```
make test     # unit and integration suite
make smoke    # every stage at reduced size into outputs/_smoke
make all      # the full pipeline
```

Runtimes below are for the machine the thesis ran on, a 2020 MacBook Air (M1), where
`make test` takes about 30 s, `make smoke` a few minutes and `make all` about 16 h. The work
runs on one core per stage, so expect times to scale with single-core speed.

## Inputs

Both files come from one Zenodo record; the top-level README gives the download commands.
`00_check_data.py` reports what is missing and where it goes.

| file | what it is |
|---|---|
| `data/Prior.csv` | LOVECLIM transient run, long format (lon, lat, age, mtco, mtwa) |
| `data/Observation.csv` | Liu et al. (2026) fxTWAPLS pollen reconstructions |

The cube is parsed once and cached under `data/cache/`, so only the first run pays for it.

## Order and cost

The order is forced by what each stage inherits. `01` selects the regularizer that every
analog estimator holds fixed, so it runs first; `06` needs `04`'s operating point; the
results notebook reads all of them.

| # | stage | ~runtime (M1) | inherits from |
|---|---|---|---|
| 00 | `00_check_data.py` | seconds | |
| 01 | `01_3dvar.py` | **~3.8 h** | |
| 02 | `02_hgaoenkf.py` | ~1.6 h | 01 |
| 03 | `03_hgaoenkf_evidence.py` | ~2.0 h | 01 |
| 04 | `04_hgaoenkf_mt.py` | **~6.5 h** | 01 |
| 05 | `05_ablations.py` | ~10 min | 01 |
| 06 | `06_product.py` | ~25 min | 04 |
| | `notebooks/final_results.ipynb` (`make figures`) | ~1 min | 01-06 |

Stages 02, 03, 04 and 05 depend only on 01, so they can run concurrently given the cores.
Runtimes are extrapolated from measured per-call costs and are good to about +-30% on that
machine.

Every stage prints a linear-rate ETA inside its loops, and a `[3/5]` banner between them.

## Re-running one stage

Every script takes `--only <stage>[,<stage>]`, which runs that stage alone against the
operating points already stored. The stage names are the ones each script lists when it
starts; an unrecognised name is refused rather than silently running nothing.

`make trajectories` is the common case: it re-runs the trajectory lane for all four
estimators and then the figures, about an hour on the M1, leaving every grid untouched.

## Threading

`_common.py` caps BLAS to one thread before numpy loads. The assimilation is thousands of
small eigendecompositions, a size where splitting each across cores costs more than it
saves; unpinned, the analog estimators run several times slower. Nothing can undo the cap
once numpy is imported, which is why every script imports `_common` first.

## What gets written

```
outputs/
  runs/<estimator>/{ppe,trajectory,withholding}/   metrics.csv, analysis npz, config.json
  ablations/{analog_taper_*,mt_stack,rep_var}/     the same schema, one directory per question
  product/{main,no_flow,exclude_0,...}/            reconstruction fields and config
  figures/report/                                  the figures the report prints
  tables/                                          printed tables, as CSV
```

`paleoreco/paths.py` is the only module that names these, so moving one is a single edit.

## Reading the results

`notebooks/final_results.ipynb` holds every figure, table and number the report prints, in
report order. It reads these directories and the raw inputs, runs no estimator, and writes the
report figures to `outputs/figures/report/`. `make figures` executes it in place.
