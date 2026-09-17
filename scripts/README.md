# Pipeline

Sparse paleoclimate field reconstruction by data assimilation: three evaluation lanes
over four estimators, the reconstruction they select, and the report figures.

```
make test     # unit and integration suite, ~30 s
make smoke    # every stage at reduced size into outputs/_smoke, a few minutes
make all      # the full pipeline, ~15-16 h
```

## Inputs

Neither raw file can be fetched automatically; `00_check_data.py` reports what is
missing and where it goes.

| file | what it is |
|---|---|
| `data/Prior.csv` | LOVECLIM transient run, long format (lon, lat, age, mtco, mtwa) |
| `data/Observation.csv` | Liu et al. (2026) fxTWAPLS pollen reconstructions |

The cube is parsed once and cached under `data/cache/`, so only the first run pays for it.

## Order and cost

The order is forced by what each stage inherits. `01` selects the regularizer that every
analog estimator holds fixed, so it runs first; `06` needs `04`'s operating point; `07`
reads all of them.

| # | stage | ~runtime | inherits from |
|---|---|---|---|
| 00 | `00_check_data.py` | seconds | |
| 01 | `01_3dvar.py` | **~3.8 h** | |
| 02 | `02_hgaoenkf.py` | ~1.6 h | 01 |
| 03 | `03_hgaoenkf_evidence.py` | ~2.0 h | 01 |
| 04 | `04_hgaoenkf_mt.py` | **~6.5 h** | 01 |
| 05 | `05_ablations.py` | ~10 min | 01 |
| 06 | `06_product.py` | ~15 min | 04 |
| 07 | `07_figures.py` | ~10 min | 01, 02, 04, 06 |

Stages 02, 03, 04 and 05 depend only on 01, so they can run concurrently given the cores.
Runtimes are extrapolated from measured per-call costs and are worth about +-30%.

Every stage prints a linear-rate ETA inside its loops, and a `[3/5]` banner between them.

## Re-running one stage

Every script takes `--only <stage>[,<stage>]`, which runs that stage alone against the
operating points already stored. The stage names are the ones each script lists when it
starts; an unrecognised name is refused rather than silently running nothing.

`make trajectories` is the common case: it re-runs the trajectory lane for all four
estimators and then the figures, about an hour, leaving every grid untouched.

## Threading

`_common.py` caps BLAS to one thread before numpy loads. The assimilation is thousands of
small eigendecompositions, a size where splitting each across cores costs more than it
saves; unpinned, the analog estimators run several times slower. Nothing can undo the cap
once numpy is imported, which is why every script imports `_common` first.

## What gets written

```
outputs/
  runs/<estimator>/{ppe,trajectory,withholding}/   metrics.csv, analysis npz, config.json
  ablations/{analog_taper,mt_stack,rep_var}/       the same schema, one directory per question
  product/{main,no_flow,exclude_0,...}/            reconstruction fields and config
  figures/report/                                  the figures the report prints
  tables/                                          printed tables, as CSV
```

`paleoreco/paths.py` is the only module that names these, so moving one is a single edit.

## Reading the results

`notebooks/results_summary.ipynb` reads these directories and nothing else. It writes
nothing that ships: anything the report prints is produced by `07_figures.py`.
