# ocean-ml-closure

A Julia pipeline that learns ocean surface forcing (wind stress and heat
flux) from ERA5 atmospheric reanalysis using a small MLP, then feeds those
learned fluxes into a 1-D Oceananigans ocean column at 6 globally
distributed sites and compares the result to GLORYS12 ocean reanalysis.

```
ERA5 atmosphere  →  MLP surface-flux model  →  Oceananigans 1-D column  →  ocean response vs GLORYS
```

For the full write-up see (docs/project_summary.md).

## What's here

- **ML side:** v1 (8 inputs, small net) and v2 (13 inputs with engineered
  features + real humidity, bigger net with cosine LR). Each trained with a
  linear-regression baseline for comparison. Temporal 80/20 split to avoid
  leakage.
- **Sim side:** Oceananigans 1-D columns driven by the MLP's predicted
  surface fluxes. Four configurations per site: v1 ML, v2 ML (paper
  headline), no-QT (wind stress only, baseline), and v3 bulk-formula
  ceiling experiment.
- **Notebooks:** 7 analysis notebooks covering sim evolution, animations,
  ML scatter, MLD, forcing timeseries, regime analysis, and the v3 ceiling
  finding.

## Headline results

v2 MLP offline skill (RMSE / std of target, temporal held-out):

| Target | v1 MLP | **v2 MLP** |
| τx     | 0.33   | **0.10**   |
| τy     | 0.35   |   **0.10** |
| QT     |   0.20 |   **0.12** |

Coupled sim, day-30 surface T vs GLORYS at 6 sites: between 0.04 °C (best,
lat30/-50) and 1.66 °C (worst, lat0/-140).

**Key finding from the ceiling experiment:** replacing the MLP with the
analytic bulk formula as the sim's forcing changes the day-30 surface T by
less than 0.02 °C at every site. The ML is not the bottleneck; the 1-D
simulator's architecture is.

## Repository layout

```
src/
  forcing/       ERA5 loaders, bulk flux formulas, dataset builders (v1 + v2)
  ml/            MLP and linear trainers (v1 + v2)
  sims_multi/    Oceananigans sims (v1, v2, v3 with bulk-mode toggle)
  analysis/      GLORYS processing

notebooks/       7 Jupyter notebooks using the Julia kernel (IJulia)
data/            ERA5, GLORYS, processed datasets (raw files gitignored)
output/era5/     all sim outputs (gitignored)
docs/            project summary
memory/          working notes, paper references (gitignored)
```

## Running the pipeline

### Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# for notebooks (one-time)
julia --project=. -e 'using IJulia; IJulia.installkernel("Julia")'
```

### ERA5 (raw files not in repo, download separately)

Six ERA5 files, one per site, with variables `u10 v10 t2m sst d2m ssrd strd`,
(30 DAYS min). Drop into `data/NewEra5/<site>.nc`, e.g.
`data/NewEra5/lat30lon-50.nc`.

### GLORYS (Copernicus, not in repo, download separately)

Six daily GLORYS12 files (`cmems_mod_glo_phy_my_0.083deg_P1D-m`) with
variables `uo vo so thetao`, (30 days min). Drop into
`data/Copernicus_<site>/*.nc`.

### Process GLORYS

```bash
julia --project=. src/analysis/process_glorys.jl lat30lon-50
# repeat for each site
```

### Build dataset + train (v2)

```bash
julia --project=. src/forcing/build_era5_multisite_forcing_dataset_v2.jl
julia --project=. src/ml/train_era5_multisite_forcing_models_v2.jl
```

### Run coupled sims at any site

```bash
# v2 ML-forced run (headline)
SITE=lat30lon-50 julia --project=. src/sims_multi/ml_forced_run_v2.jl

# v2 no-QT baseline
SITE=lat30lon-50 julia --project=. src/sims_multi/ml_forced_run_noqt_v2.jl

# v3 with bulk-formula ceiling experiment
SITE=lat30lon-50 USE_BULK=1 julia --project=. src/sims_multi/ml_forced_run_v3.jl
```

Outputs in `output/era5/`.

### Figures

Open any notebook in `notebooks/` with a Julia 1.10 kernel.

## Limitations

- 1-D column only; no lateral advection or mesoscale eddies
- Bulk eddy viscosity, not a real turbulence closure (KPP/CATKE)
- Multi-site validation is in-distribution (all test sites were also trained
  on). Leave-one-site-out generalization is future work.

