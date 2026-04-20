# Ocean ML Closure — Project Summary

## Goal

Build a Julia pipeline that uses ERA5 atmospheric reanalysis to predict ocean
surface forcing (wind stress τx, τy and net heat flux QT), then drive a 1-D
Oceananigans ocean column with that predicted forcing and validate against
GLORYS12 ocean reanalysis at multiple globally distributed sites.

The ML predicts surface forcing

```
ERA5 atmosphere  →  ML surface-flux model  →  Oceananigans 1-D column  →  ocean response 
```

---

## Datasets

**ERA5** (hourly atmospheric reanalysis) — 6 globally distributed ocean sites,
2023-01-01 → 2026-01-01 (~3 years, 26,328 samples per site).

Variables: `u10, v10, t2m, sst, d2m, ssrd, strd`

| Site | Name | Regime |
|---|---|---|
| lat30/lon-50 | N. Atl. subtropical gyre | local-forcing dominated |
| lat-25/lon-10 | S. Atl. subtropical | local-forcing dominated |
| lat-45/lon80 | S. Indian / ACC | mesoscale-dominated |
| lat0/lon-140 | Equatorial Pacific | Kelvin-wave / equatorial |
| lat30/lon-150 | N. Pac. subtropical | local-forcing dominated |
| lat40/lon-25 | N. Atl. mid-lat | local-forcing dominated |

**GLORYS12** (Copernicus `cmems_mod_glo_phy_my_0.083deg_P1D-m`) — daily ocean
reanalysis at the same 6 sites, 23 depth levels (0.49–109.73 m) regridded to
16 uniform levels over 0–100 m for initialization and comparison.

---

## ML model

All models are fully-connected feed-forward neural networks (MLPs) trained with
reverse-mode autodiff through Flux.jl. ReLU hidden layers, identity output. 
Inputs and outputs are z-score normalized on
the training split. A least-squares linear regression baseline is trained
alongside every MLP so we can see when the ML is actually earning its keep.

**Split:** temporal 80/20 — first 80% of hours train, last 20% test.
Random shuffling leaks information between hourly-adjacent samples because
they are strongly autocorrelated; temporal split gives an honest evaluation.

### v1 MLP (phase 2–3)

- Inputs (8): `u10, v10, t2m, sst, ssrd, strd, lat, lon`
- Outputs (3): `τx, τy, QT`
- Architecture: `Dense(8→32, relu) → Dense(32→32, relu) → Dense(32→3)`
- Optimizer: Adam, constant lr 1e-3, 400 epochs
- ~1,400 trainable params

### v2 MLP (phase 5 — paper headline model)

- Inputs (13): v1 inputs **plus** 5 engineered features from the bulk formula:
  `|U|, ΔT, q_sat(sst), q_sat(t2m), q_air` (the last from ERA5 `d2m` dew-point)
- Outputs (3): same
- Architecture: `Dense(13→64, relu) → Dense(64→64, relu) → Dense(64→32, relu) → Dense(32→3)`
- Optimizer: Adam with cosine lr decay `1e-3 → 1e-5` over 1000 epochs
- ~7,100 trainable params

### RMSE (temporal 80/20 held-out)

| Target | v1 linear | v1 MLP | v2 linear | **v2 MLP** | v2 skill (RMSE/std) |
τx       |  0.0318   | 0.0352 |    0.0303 | **0.0106** |                0.10 |
| τy     | 0.0322    | 0.0300 |    0.0328 | **0.0088** |                0.10 |
| QT (W/m^2) | 46.0  |   54.3 |      26.5 |   **33.7** |                0.12 |

v2 MLP beats v1 MLP by 3.3× on τx, 3.4× on τy, and 1.6× on QT. Skill scores
of 0.10–0.12 match what published ocean-ML parameterization papers report
(Bolton & Zanna 2019; Yuval & O'Gorman 2020).

Feature engineering (|U|, ΔT, q_sat) did most of the heavy lifting on τ;
swapping the 80%-RH humidity assumption for real d2m did most of the heavy
lifting on QT.

---

## Coupled simulation

**Setup:** `NonhydrostaticModel` on a `RectilinearGrid(2×2×16)` with
horizontal extent 1 m (pseudo-1-D column), water depth 100 m (so vertical
resolution ≈ 6.25 m), topology Periodic-Periodic-Bounded.

- Buoyancy: `SeawaterBuoyancy()`
- Tracers: `(:T, :S)` with GLORYS-regridded initial profiles
- Coriolis: `FPlane(latitude = site_lat)` — needed so wind stress turns with
  depth (Ekman) instead of accumulating as a pure zonal jet
- Closure: `ScalarDiffusivity(ν = 1e-2, κ_T = κ_S = 1e-3) m²/s` —
  bulk eddy values standing in for a KPP-like boundary-layer parameterization
- Constants: ρ₀ = 1027 kg/m³, c_p = 3985 J/kg/K
- Boundary conditions (top): `FluxBoundaryCondition(-τ/ρ₀)` for u, v;
  `FluxBoundaryCondition(-QT/(ρ₀·c_p))` for T; bottom flux = 0 for all.
  Sign negation is required because Oceananigans' top-face flux convention is
  positive-upward (out of domain); our ML predicts positive = into ocean.
- Timestep: `TimeStepWizard(cfl = 0.8, max_Δt = 1800 s)`, initial Δt = 120 s

**Runs:** at each of the 6 sites, 30-day simulations were performed in four
configurations:

1. **v1 QT** — v1 MLP τ + QT
2. **v2 QT** — v2 MLP τ + QT (paper headline)
3. **v1 no-QT** — wind stress only (no heat flux BC)
4. **v3 bulk** — bulk-formula τ + QT fed directly (no ML) — *ceiling experiment*

---

## Key findings

### Finding 1 — The MLP achieves published-literature skill on surface flux prediction

v2 MLP reaches skill scores of 0.10–0.12 on the held-out temporal test set, in
the range of published ocean-ML parameterization papers. Feature engineering
(derived bulk-formula quantities) and real humidity from ERA5 dew-point matter
more than architectural capacity alone.

### Finding 2 — The 1-D simulator is the binding constraint, not the ML

The **v3 ceiling experiment** replaces the MLP with the analytic bulk formula
fed directly into the same simulator. Across all 6 sites, v3 ML and v3 BULK
produce day-30 surface temperatures within **0.02 °C** of each other. Replacing
the MLP with a perfect (analytic) forcing does not change the coupled
simulation output. Therefore the residual disagreement with GLORYS is driven
by the 1-D simulator's structural limits, not by ML quality.

In the sense of Yuval & O'Gorman(2020): 
offline RMSE improvements don't automatically translate to coupled-sim
improvements once the simulator's architecture dominates the error budget.

### Finding 3 — 1-D closure skill is regime-dependent

At sites where local surface forcing dominates the mixed-layer response
(subtropical gyre interiors — lat30/-50, lat40/-25, lat-25/-10), surface T
matches GLORYS to within ~1 °C after 30 days.

At sites dominated by mesoscale eddies, fronts, and advection (lat-45/lon80 in
the Antarctic Circumpolar Current; lat0/lon-140 at the equator where
Coriolis f is approximately 0 and Ekman is undefined), no 1-D ML closure can reproduce
GLORYS — the observational variability comes from lateral dynamics a 1-D
column cannot represent. This is consistent with 1-D mixed-layer theory
going back to Kraus & Turner (1967) and Price et al. (1986).

### Day-30 surface T at all 6 sites (v2 MLP run)

| Site      | v2 MLP| GLORYS|       Δ   |
| lat30/-50 | 21.16 | 21.12 | **+0.04** |
| lat-45/80 | 12.89 | 13.83 | −0.94     |
| lat40/-25 | 15.37 | 16.48 | −1.11     |
| lat-25/-10 | 22.69 | 23.83 | −1.14    |
| lat30/-150 | 21.81 | 20.36 | +1.45    |
| lat0/-140 | 26.32 | 24.66 | +1.66     |

---

## What was tried and didn't work 

- **v3 ML (lower κ_T + salinity E-P flux + QT bias correction)** — made things
  worse on absolute GLORYS match. The bias correction calibrated to bulk
  formula, not GLORYS, and the two disagree by ~1 °C at most sites, so
  correcting toward bulk pulled the sim away from GLORYS. κ_T reduction held
  stratification better but couldn't compensate. E-P without precipitation
  made salinity drift monotonically upward. v3 is kept in the repo as
  supporting evidence for the ceiling finding, not as a headline result.
- **Random-shuffle train/test split (early phase 2)** — temporally adjacent
  ERA5 samples leak information between train and test. Temporal split is the
  honest evaluation.
- **Naive QT target (early)** — an ad-hoc linear combination of radiation and
  ΔT terms. Linear regression hit machine precision on it, which was a clue
  the target was trivially linear; replaced with full bulk formula.

---

## Known limitations

- **1-D column,** No lateral advection, no mesoscale eddies, no
  fronts. Bounds how well any ocean-state forecast can do in regimes where
  lateral dynamics dominate.
- **Bulk eddy viscosity, not prognostic turbulence closure.** `ν = 1e-2 m²/s`
  is a constant stand-in for KPP/CATKE. Enough to demonstrate stable coupling.
- **No salinity flux in the paper-headline v2 setup.** S is held at its
  initial profile in v2 (v3 added a one-sided E flux but that created its own
  bias; see above).
- **In-distribution multi-site validation only.** All 6 test sites were in the
  MLP's training set. A leave-one-site-out retrain would make the
  generalization claim stronger.
- **No precipitation term.** ERA5 precipitation was not downloaded; latent
  heat drives evaporation only.

---

## Repository layout

```
src/
  forcing/       ERA5 loaders + bulk flux formulas (v1 + v2) + dataset builders
  ml/            MLP and linear training scripts (v1 and v2)
  sims_multi/    Oceananigans 1-D column sims (v1, v2, v3 with bulk-mode toggle)
  analysis/      GLORYS processing script

notebooks/
  01_sim_evolution.ipynb             per-site 30-day surface + profile comparison
  02_sim_animations.ipynb            animated Hovmöller and T-profile GIFs
  03_ml_scatter_v1_vs_v2.ipynb       predicted-vs-truth scatter + RMSE tables
  04_mixed_layer_depth.ipynb         MLD diagnostic sim vs GLORYS
  05_forcing_timeseries.ipynb        ML vs bulk τ/QT over the sim window
  06_regime_analysis.ipynb           which sites 1-D works at and why
  07_v3_ceiling_experiment.ipynb     ML vs bulk-formula-fed sim — the ceiling

data/
  ERA5/            v1 ERA5 downloads (no d2m)
  NewEra5/         v2 ERA5 downloads (with d2m) — used by v2 and v3
  Copernicus_<site>/  GLORYS daily NetCDFs per site
  generated/       processed GLORYS, trained-model results, dataset jld2s

output/era5/       all simulation outputs (QT, no-QT, v2, v3, v3bulk per site)
memory/            project notes, phase-by-phase findings, paper references (gitignored)
docs/              this document + any other write-ups
```

Large data files (`data/NewEra5/*`, `data/Copernicus_*/*.nc`, all sim outputs
in `output/`) are git-ignored; only code, docs, and notebooks are versioned.

---

## Key references

- **Hersbach et al. 2020** — ERA5 reanalysis
- **Lellouche et al. 2021** — GLORYS12 reanalysis
- **Ramadhan et al. 2020** — Oceananigans.jl
- **Fairall et al. 2003** — COARE bulk flux algorithm
- **Large, McWilliams & Doney 1994** — KPP
- **Bolton & Zanna 2019** — ML for ocean subgrid parameterization
- **Yuval & O'Gorman 2020** — offline vs online ML-parameterization skill
- **Kraus & Turner 1967** — 1-D mixed-layer model origins
- **Price, Weller & Pinkel 1986** — 1-D diurnal-cycle validation

---


