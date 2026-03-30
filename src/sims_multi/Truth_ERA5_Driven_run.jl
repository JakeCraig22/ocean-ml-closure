using Oceananigans
using Oceananigans: TimeInterval, IterationInterval
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using Oceananigans.Forcings: ContinuousForcing
using Printf

include(joinpath(@__DIR__, "..", "forcing", "build_forcing_timeseries.jl"))
include(joinpath(@__DIR__, "..", "common", "snapshot_io.jl"))

using .BuildForcingTimeseries: build_forcing_series, interp1
using .CommonSnapshotIO: save_snapshot_factory

era5_path = raw"data/ERA5/reanalysis-era5-single-levels-timeseries-sfc6369p5v8.nc"

forcing = build_forcing_series(era5_path)

water_depth = 100.0
vert_res = 256
Δt0 = 1.0
stop_time = forcing.time[end]

grid = RectilinearGrid(
    size = (8, 8, vert_res),
    extent = (1, 1, water_depth),
    topology = (Periodic, Periodic, Bounded)
)

τx_fun(x, y, z, t) = interp1(t, forcing.time, forcing.τx) / 1027.0
τy_fun(x, y, z, t) = interp1(t, forcing.time, forcing.τy) / 1027.0

u_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(τx_fun),
    bottom = FluxBoundaryCondition(0.0)
)

v_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(τy_fun),
    bottom = FluxBoundaryCondition(0.0)
)

κ_tracers = (T = 1e-5, S = 1e-5)

model = NonhydrostaticModel(
    grid;
    buoyancy = SeawaterBuoyancy(),
    tracers = (:T, :S),
    closure = (AnisotropicMinimumDissipation(),
               ScalarDiffusivity(ν = 1e-4, κ = κ_tracers)),
    boundary_conditions = (u = u_bcs, v = v_bcs)
)

set!(model,
    T = (x, y, z) -> -z / water_depth,
    S = (x, y, z) -> 35.0
)

wizard = TimeStepWizard(cfl = 0.8, max_Δt = 10.0)
simulation = Simulation(model, Δt = Δt0, stop_time = stop_time)

add_callback!(simulation, wizard, IterationInterval(1))

outdir = "output/era5/truth_Nz256"
save_snapshot = save_snapshot_factory(outdir)
add_callback!(simulation, save_snapshot, TimeInterval(300))

println("Running ERA5 truth case")
run!(simulation)