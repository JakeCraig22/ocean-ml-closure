using Oceananigans
using Oceananigans: TimeInterval, IterationInterval
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using JLD2
using Flux

# no-QT variant of ml_forced_run.jl — wind stress only, skip the heat flux BC.
# used as a baseline to show how much QT actually does for the sim.

include(joinpath(@__DIR__, "..", "forcing", "load_era5.jl"))
include(joinpath(@__DIR__, "..", "forcing", "build_forcing_timeseries.jl"))

using .LoadERA5: load_era5_point
using .BuildForcingTimeseries: interp1

const SITES = Dict(
    "lat30lon-50"  => (lat= 30.0, lon= -50.0),
    "lat-25lon-10" => (lat=-25.0, lon= -10.0),
    "lat-45lon80"  => (lat=-45.0, lon=  80.0),
    "lat0lon-140"  => (lat=  0.0, lon=-140.0),
    "lat30lon-150" => (lat= 30.0, lon=-150.0),
    "lat40lon-25"  => (lat= 40.0, lon= -25.0),
)
site = get(ENV, "SITE", "lat30lon-50")
haskey(SITES, site) || error("Unknown site $site")
site_lat = SITES[site].lat
site_lon = SITES[site].lon

era5_file   = "data/ERA5/$(site)ERA5.nc"
model_file  = "data/generated/era5_forcing_results_multisite.jld2"
glorys_file = site == "lat30lon-50" ?
    "data/generated/glorys_processed.jld2" :
    "data/generated/glorys_processed_$(site).jld2"

out_file = "output/era5/ml_forced_30day_glorysinit_noqt_$(site).jld2"
tmp_file = "output/era5/ml_forced_30day_glorysinit_noqt_$(site)_checkpoint.jld2"

water_depth = 100.0
vert_res    = 16
sim_days = 30
Δt0    = 120.0
max_Δt = 1800.0
ρ0     = 1027.0
ν_eddy = 1e-2
κ_eddy = 1e-3
save_interval = 24 * 3600.0

forcing_data = load_era5_point(era5_file)
u10, v10, t2m, sst, ssrd, strd, era5_time =
    forcing_data.u10, forcing_data.v10, forcing_data.t2m, forcing_data.sst,
    forcing_data.ssrd, forcing_data.strd, forcing_data.time

Nt = length(era5_time)
stop_time = min(era5_time[end], sim_days * 86400.0)

d = JLD2.load(model_file)
ml_model = d["model"]
xμ, xσ, yμ, yσ = d["xμ"], d["xσ"], d["yμ"], d["yσ"]

glorys = JLD2.load(glorys_file)
T_init_profile = Float64.(glorys["T_profiles"][1])
S_init_profile = Float64.(glorys["S_profiles"][1])
depth_profile  = Float64.(glorys["target_depth"])

println("Loaded GLORYS initial profiles")
println("GLORYS T init min/max = ", minimum(T_init_profile), " / ", maximum(T_init_profile))

# same inference loop as the QT version but only keep τx/τy outputs
τx = zeros(Float64, Nt)
τy = zeros(Float64, Nt)

for i in 1:Nt
    x = Float64[u10[i], v10[i], t2m[i], sst[i], ssrd[i], strd[i], site_lat, site_lon]
    x_norm = (x .- xμ) ./ xσ
    y_norm = vec(ml_model(Float32.(x_norm)))
    y = y_norm .* yσ .+ yμ

    τx[i] = y[1]
    τy[i] = y[2]
    # skip y[3] on purpose — that's the QT this run doesn't use
end

println("τx stats: min=", minimum(τx), " max=", maximum(τx))
println("τy stats: min=", minimum(τy), " max=", maximum(τy))

grid = RectilinearGrid(size=(2, 2, vert_res), extent=(1.0, 1.0, water_depth),
                      topology=(Periodic, Periodic, Bounded))

# negate for oceananigans top-flux convention (same reason as qt version)
τx_fun(_, _, t) = -interp1(t, era5_time, τx) / ρ0
τy_fun(_, _, t) = -interp1(t, era5_time, τy) / ρ0

u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τx_fun), bottom=FluxBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τy_fun), bottom=FluxBoundaryCondition(0.0))

κ_tracers = (T = κ_eddy, S = κ_eddy)

# no T BC passed — just wind stress. T will drift from pure advection/diffusion only
model = NonhydrostaticModel(
    grid;
    coriolis = FPlane(latitude = site_lat),
    buoyancy = SeawaterBuoyancy(),
    tracers = (:T, :S),
    closure = ScalarDiffusivity(ν = ν_eddy, κ = κ_tracers),
    boundary_conditions = (u = u_bcs, v = v_bcs)
)

function interp_profile(depth_val, depth_grid, profile)
    depth_val <= depth_grid[1]   && return profile[1]
    depth_val >= depth_grid[end] && return profile[end]
    j = searchsortedlast(depth_grid, depth_val)
    j = min(j, length(depth_grid) - 1)
    d0, d1 = depth_grid[j], depth_grid[j + 1]
    p0, p1 = profile[j], profile[j + 1]
    return p0 + (p1 - p0) * (depth_val - d0) / (d1 - d0)
end

T_init(_, _, z) = interp_profile(-z, depth_profile, T_init_profile)
S_init(_, _, z) = interp_profile(-z, depth_profile, S_init_profile)
set!(model, T = T_init, S = S_init)

simulation = Simulation(model, Δt = Δt0, stop_time = stop_time)
wizard = TimeStepWizard(cfl = 0.8, max_Δt = max_Δt)
add_callback!(simulation, wizard, IterationInterval(1))

function progress(sim)
    println("progress: day=", round(sim.model.clock.time / 86400, digits=2),
            " / ", round(stop_time / 86400, digits=2),
            " | iter=", iteration(sim),
            " | dt=", sim.Δt)
end
add_callback!(simulation, progress, TimeInterval(86400))

mkpath(dirname(out_file))

saved_times = Float64[]
u_profiles  = Vector{Vector{Float64}}()
v_profiles  = Vector{Vector{Float64}}()
T_profiles  = Vector{Vector{Float64}}()
S_profiles  = Vector{Vector{Float64}}()

function write_checkpoint()
    @save tmp_file saved_times u_profiles v_profiles T_profiles S_profiles τx τy era5_time site_lat site_lon water_depth vert_res Δt0 max_Δt save_interval era5_file model_file glorys_file
end

function save_profiles!(sim)
    push!(saved_times, sim.model.clock.time)
    push!(u_profiles, copy(vec(Array(interior(sim.model.velocities.u))[1, 1, :])))
    push!(v_profiles, copy(vec(Array(interior(sim.model.velocities.v))[1, 1, :])))
    push!(T_profiles, copy(vec(Array(interior(sim.model.tracers.T))[1, 1, :])))
    push!(S_profiles, copy(vec(Array(interior(sim.model.tracers.S))[1, 1, :])))
    write_checkpoint()
    println("saved checkpoint at day=", round(sim.model.clock.time / 86400, digits=2),
            " | num_saves=", length(saved_times))
end

add_callback!(simulation, save_profiles!, TimeInterval(save_interval))
save_profiles!(simulation)

println("Running $(sim_days)-day ML-forced run (wind stress only, no QT)")
run!(simulation)

if isempty(saved_times) || saved_times[end] != model.clock.time
    save_profiles!(simulation)
end

@save out_file saved_times u_profiles v_profiles T_profiles S_profiles τx τy era5_time site_lat site_lon water_depth vert_res Δt0 max_Δt save_interval era5_file model_file glorys_file

println("$(sim_days)-day no-QT run completed successfully.")
println("Saved final output to: ", out_file)
