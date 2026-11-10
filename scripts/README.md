# Pipeline

Sparse paleoclimate field reconstruction by data assimilation: three evaluation lanes over
four estimators, the reconstruction they select, and the report figures.

```
make test     # unit and integration suite
make smoke    # every stage at reduced size into outputs/_smoke, then the notebook on it
make all      # the full pipeline
```

On a 2020 MacBook Air (M1), `make test` takes a few seconds and `make smoke` five to eight
minutes. The full run's time is pending the full re-run. The work runs on one core per stage, so
expect times to scale with single-core speed.

## Inputs

Both files come from one Zenodo record; the top-level README gives the download commands.
`00_check_data.py` reports what is missing and where it goes.

| file | what it is |
|---|---|
| `data/Prior.csv` | LOVECLIM transient run, long format (lon, lat, age, mtco, mtwa) |
| `data/Observation.csv` | Liu et al. (2026) fxTWAPLS pollen reconstructions |

The cube is parsed once and cached under `data/cache/`, so only the first run pays for it.

## Stages

Each stage owns one question and reads what it inherits from the configs earlier stages
stored, so the order is forced: 3DVar's snapshot run fixes the taper every estimator uses,
evidence-only's snapshot run fixes the κ MTA-HGAOEnKF uses, and MTA-HGAOEnKF's snapshot run
fixes the product.

| # | script | stages (`--only` names) | inherits |
|---|---|---|---|
| 00 | `00_check_data.py` | | |
| 01 | `01_3dvar.py` | `taper-grid`, `snapshot`, `time-series`, `withholding`, `obs-model` | |
| 02 | `02_evidence.py` | `evidence-grid`, `snapshot`, `correlation-check`, `time-series`, `withholding` | 01's taper |
| 03 | `03_mta.py` | `mta-grid`, `snapshot`, `time-series`, `withholding`, `ladder` | 01's taper, 02's κ |
| 04 | `04_product.py` | `product` | 03's operating point |
| | `notebooks/final_results.ipynb` (`make figures`) | | 01 to 04 |

What each one does:

- **01, 3DVar.** The taper grid scans localization L, shrinkage λ and coupling γ jointly with
  the background amplitude c on the snapshot lane. 3DVar is then run at the selected taper on
  the snapshot, time-series and withholding lanes, and on the withholding lane once more under
  each reduced observation model (stated error alone, with the spatial term, with the temporal
  term).
- **02, evidence rule.** One grid over κ, ensemble size k, hybrid weight α and c. Its κ = 0
  slice is HGAOEnKF with the misfit rule; its best point over κ > 0 is evidence-only. Both run
  on the three lanes at their own snapshot settings. The correlation check runs the pooled and
  per-channel correlation rules at HGAOEnKF's k and α, each selecting its own c.
- **03, MTA.** One grid over the tendency share w, k, α and c at evidence-only's κ, with the
  fixed stack. MTA-HGAOEnKF runs on the three lanes at the selected point. The ladder scans the
  1-, 3- and 7-lag stacks over w at that k, α and κ.
- **04, product.** MTA-HGAOEnKF at every age of the archive, from the real network, at the
  published amplitudes.

Every grid scan is mean-only, since selection reads only rRMSE; the selected configuration is
then re-run in full. Selection is the argmin of the pooled selection-split rRMSE everywhere.

Every stage prints a linear-rate ETA inside its loops, and a `[3/5]` banner between them.

## Re-running one stage

Every script takes `--only <stage>[,<stage>]`, which runs those stages alone against the
operating points already stored. An unrecognised name is refused rather than silently running
nothing. `make trajectories` re-runs the time-series lane for all four estimators and then the
figures, leaving every grid untouched.

## Threading

`_common.py` caps BLAS to one thread before numpy loads. The assimilation is thousands of
small eigendecompositions, a size where splitting each across cores costs more than it saves;
unpinned, the analog estimators run several times slower. Nothing can undo the cap once numpy
is imported, which is why every script imports `_common` first.

## What gets written

```
outputs/
  grids/{3dvar_taper,evidence,mta}/                  metrics.csv, grid_config.json
  runs/<estimator>/{ppe,trajectory,withholding}/     metrics.csv, analysis or predictions npz,
                                                     <lane>_config.json
  ablations/correlation_rules/, mta_ladder/          a grid's metrics.csv and grid_config.json
  ablations/obs_model/{sse,sse_spatial,sse_temporal}/  3DVar withholding runs
  product/                                           reconstruction fields and config
  figures/report/                                    the figures the report prints
```

The estimators are `3dvar`, `hgaoenkf`, `hgaoenkf_evidence` and `hgaoenkf_mt`. A run's config
records the taper and analog settings it ran at, which is what the next stage reads.
`paleoreco/paths.py` is the only module that names these directories, so moving one is a single
edit; `PALEORECO_OUTPUTS` points the whole tree elsewhere.

## Reading the results

`notebooks/final_results.ipynb` holds every figure, table and number the report prints, in
report order. It reads these directories and the raw inputs, runs no estimator, and writes the
report figures to `outputs/figures/report/`. `make figures` executes it in place; `make smoke`
executes a copy against `outputs/_smoke` and writes it there.
