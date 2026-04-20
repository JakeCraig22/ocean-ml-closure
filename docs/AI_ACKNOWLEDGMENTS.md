# AI acknowledgments

AI assistance in this project = Anthropic's Claude (April 2026, Claude Code).

## AI-Assisted

- `src/sims_multi/ml_forced_run_v3.jl` (the v3 sim with bulk-mode toggle)
- `src/sims_multi/ml_forced_run_v2.jl` and `ml_forced_run_noqt_v2.jl` (v2 MLP
  inference wired into the existing sim setup)
- Adding `FPlane` Coriolis + eddy-viscosity values (`ν=1e-2`, `κ=1e-3`) to
  the sim scripts after the phase 4 debugging pointed at them
- `src/forcing/compute_surface_fluxes_v2.jl` (engineered-features function
  on top of the existing bulk formula I already had)
- `src/forcing/load_era5_v2.jl` (adds d2m to the existing loader I had)
- `src/ml/train_era5_multisite_forcing_models_v2.jl` (v2 trainer — bigger
  MLP, cosine LR decay 1e-3→1e-5) 
- `src/forcing/build_era5_multisite_forcing_dataset_v2.jl` (
  engineered features from NewEra5 into the training dataset)
- `docs/project_summary.md`
- current `README.md`
- comment cleanup across files in `src/`

## Debugging

- Sign convention on Oceananigans top-face flux BCs in the sim scripts
- Latent-heat sign in `compute_surface_fluxes.jl`
- Profile plot z-axis orientation
