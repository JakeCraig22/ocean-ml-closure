# ocean-ml-closure

This project explores a Julia-based machine learning workflow for learning ocean surface forcing from atmospheric reanalysis data, with the long-term goal of using that forcing inside Oceananigans to study ocean response.

## Project goal

The main goal of this phase of the project is to use ERA5 atmospheric and surface data to predict ocean surface forcing. Instead of directly predicting ocean motion, the model predicts the forcing applied at the ocean surface, which can later be used to drive an ocean simulation.

basically: 

**atmosphere → predicted forcing → simulated ocean response**

This makes the current work a data-driven precursor to adding in the oceananigans pkg later

## Inputs

The initial forcing model uses ERA5-derived surface and near-surface variables:

- `u10` — 10 m zonal wind
- `v10` — 10 m meridional wind
- `t2m` — 2 m air temperature
- `sst` — sea surface temperature
- `ssrd` — surface solar radiation downwards
- `strd` — surface thermal radiation downwards

In the multisite version, two additional location features are included:

- `lat` — latitude
- `lon` — longitude

So the multisite input vector is:

`[u10, v10, t2m, sst, ssrd, strd, lat, lon]`

## Targets

The model predicts three surface-forcing targets:

- `tau_x` — zonal wind stress
- `tau_y` — meridional wind stress
- `QT` — total surface heat-flux proxy

These represent the atmospheric forcing acting on the ocean surface. At this stage, the model is not directly predicting ocean motion yet. It is predicting the forcing that can later be used to drive ocean simulations.

## Phase 1: single-point proof of concept

The first phase used a small proof-of-concept dataset built from hourly ERA5 data at a single Atlantic point over roughly 16 days.

Dataset summary:

- one point
- one date range
- hourly data
- 384 total samples
- train: 307
- test: 77

The model compared a linear baseline against a small MLP.

Training summary:

- training loss dropped from about `8.08e-01` to `1.18e-03`

Phase 1 RMSE:

- `tau_x`: linear `1.574527e-02` → MLP `5.215456e-03`
- `tau_y`: linear `1.509033e-02` → MLP `4.163289e-03`
- `QT`: linear `4.720094e-13` → MLP `4.705643e-01`

Interpretation:

The MLP substantially improved prediction accuracy for `tau_x` and `tau_y` compared with the linear baseline. For `QT`, the linear model performed nearly perfectly, indicating that the current heat-flux target formulation is largely linear in the selected ERA5 inputs.

## Phase 2: multisite scaling

The second phase expanded the dataset to six different ocean locations, each covering roughly three years of hourly ERA5 data. Latitude and longitude were added as input features to provide location context.

Dataset summary:

- 6 ocean points
- about 3 years per point
- 159,384 total samples
- 8 input features
- 3 targets
- train: 127,507
- test: 31,877

MLP training summary:

- epoch 1 train MSE: `0.9151519`
- epoch 400 train MSE: `0.027845696`

Phase 2 RMSE:

- `tau_x`: linear `0.031501107` → MLP `0.01948908`
- `tau_y`: linear `0.027252417` → MLP `0.015375421`
- `QT`: linear `1.5591578e-12` → MLP `2.2722105`

Interpretation:

On the expanded 6-location ERA5 dataset with 8 input features (`u10`, `v10`, `t2m`, `sst`, `ssrd`, `strd`, `lat`, `lon`), the MLP consistently outperformed linear regression for `tau_x` and `tau_y`, while `QT` remained effectively linear. This suggests that the wind-stress targets benefit from nonlinear modeling, but the current heat-flux proxy does not yet require it.

The multisite scatter plots support this result: for `tau_x` and `tau_y`, the MLP predictions cluster more tightly around the ideal 1:1 line than the linear model. For `QT`, the linear model remains essentially perfect while the MLP is worse.

## Current takeaway

The main result so far is that machine learning helps for the wind-stress forcing targets, while the current heat-flux proxy is still dominated by linear structure.

This means the current pipeline successfully learns:

**ERA5 atmospheric state → ocean surface forcing**

and sets up the next project phase:

**predicted forcing → Oceananigans simulation → ocean response evaluation**

## Repository structure

- `src/forcing/` — ERA5 loading, target construction, and dataset builders
- `src/ml/` — training scripts for linear and MLP forcing models
- `src/analysis/` — plotting and result visualization scripts
- `notebooks/` — inspection and summary notebooks
- `envs/forcing_only/` — lightweight Julia environment used for forcing-learning experiments
- `data/generated/` — generated datasets, trained-model results, and plots

## Running the current workflow

Use the lightweight forcing environment for the forcing-learning phase.

### Build the single-point dataset

```bash
julia --project=envs/forcing_only src/forcing/build_era5_forcing_dataset.jl
