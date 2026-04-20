using Oceananigans
using Oceananigans: TimeInterval, IterationInterval
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using JLD2
using Flux

# v2 sim. same sim setup as v1 but uses the v2 mlp (trained on NewEra5 w/ d2m,
# 13 inputs, bigger net). everything else is identical: eddy viscosity, fplane
# coriolis, negated flux BCs for oceananigans convention.
# site via ENV["SITE"], sim_days via ENV["SIM_DAYS"] (default 30).

include(joinpath(@__DIR__, "..", "forcing", "load_era5_v2.jl"))
include(joinpath(@__DIR__, "..", "forcing", "compute_surface_fluxes_v2.jl"))
include(joinpath(@__DIR__, "..", "forcing", "build_forcing_timeseries.jl"))

using .LoadERA5v2: load_era5_point_v2
using .ComputeSurfaceFluxesV2: engineered_features
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

era5_file   = "data/NewEra5/$(site).nc"
model_file  = "data/generated/era5_forcing_results_v2.jld2"
glorys_file = site == "lat30lon-50" ?
    "data/generated/glorys_processed.jld2" :
    "data/generated/glorys_processed_$(site).jld2"

water_depth = 100.0
vert_res    = 16

sim_days = parse(Int, get(ENV, "SIM_DAYS", "30"))

out_file = "output/era5/ml_forced_$(sim_days)day_v2_qt_$(site).jld2"
tmp_file = "output/era5/ml_forced_$(sim_days)day_v2_qt_$(site)_checkpoint.jld2"

Δt0    = 120.0
max_Δt = 1800.0
ρ0     = 1027.0
c_p    = 3985.0
ν_eddy = 1e-2
κ_eddy = 1e-3
save_interval = 24 * 3600.0

# --- load era5 (v2 — has d2m) ---
data = load_era5_point_v2(era5_file)
u10, v10, t2m, sst, d2m, ssrd, strd, era5_time =
    data.u10, data.v10, data.t2m, data.sst, data.d2m, data.ssrd, data.strd, data.time

Nt = length(era5_time)
stop_time = min(era5_time[end], sim_days * 86400.0)

# --- load v2 model ---
d = JLD2.load(model_file)
ml_model = d["model"]
xμ, xσ, yμ, yσ = d["xμ"], d["xσ"], d["yμ"], d["yσ"]

# --- glorys init ---
glorys = JLD2.load(glorys_file)
T_init_profile = Float64.(glorys["T_profiles"][1])
S_init_profile = Float64.(glorys["S_profiles"][1])
depth_profile  = Float64.(glorys["target_depth"])

println("[v2] site=$site  lat=$site_lat lon=$site_lon")
println("GLORYS T init min/max = ", minimum(T_init_profile), " / ", maximum(T_init_profile))
println("GLORYS S init min/max = ", minimum(S_init_profile), " / ", maximum(S_init_profile))

# --- compute engineered features once for all times, then run mlp hour by hour ---
eng = engineered_features(u10, v10, t2m, sst, d2m)   # Nt × 5

τx = zeros(Float64, Nt)
τy = zeros(Float64, Nt)
QT = zeros(Float64, Nt)

for i in 1:Nt
    # 8 raw + 5 engineered = 13 inputs, same order the v2 net was trained on
    x = Float64[
        u10[i], v10[i], t2m[i], sst[i], ssrd[i], strd[i], site_lat, site_lon,
        eng[i, 1], eng[i, 2], eng[i, 3], eng[i, 4], eng[i, 5],
    ]
    x_norm = (x .- xμ) ./ xσ
    y_norm = vec(ml_model(Float32.(x_norm)))
    y = y_norm .* yσ .+ yμ

    τx[i] = y[1]
    τy[i] = y[2]
    QT[i] = y[3] / (ρ0 * c_p)   # W/m² → K·m/s
end

println("τx  stats: min=", minimum(τx),  " max=", maximum(τx))
println("τy  stats: min=", minimum(τy),  " max=", maximum(τy))
println("QT  stats (W/m²):  min=", minimum(QT .* ρ0 .* c_p), " max=", maximum(QT .* ρ0 .* c_p))
println("Nt = ", Nt, " | stop_time = ", stop_time)

# --- grid + BCs (sign-negated for oceananigans top-flux convention) ---
grid = RectilinearGrid(size=(2, 2, vert_res), extent=(1.0, 1.0, water_depth),
                      topology=(Periodic, Periodic, Bounded))

τx_fun(_, _, t) = -interp1(t, era5_time, τx) / ρ0
τy_fun(_, _, t) = -interp1(t, era5_time, τy) / ρ0
QT_fun(_, _, t) = -interp1(t, era5_time, QT)

u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τx_fun), bottom=FluxBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τy_fun), bottom=FluxBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(QT_fun), bottom=FluxBoundaryCondition(0.0))

κ_tracers = (T = κ_eddy, S = κ_eddy)

model = NonhydrostaticModel(
    grid;
    coriolis = FPlane(latitude = site_lat),
    buoyancy = SeawaterBuoyancy(),
    tracers = (:T, :S),
    closure = ScalarDiffusivity(ν = ν_eddy, κ = κ_tracers),
    boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs),
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
    @save tmp_file saved_times u_profiles v_profiles T_profiles S_profiles τx τy QT era5_time site_lat site_lon water_depth vert_res Δt0 max_Δt save_interval era5_file model_file glorys_file
end

function save_profiles!(sim)
    push!(saved_times, sim.model.clock.time)
    push!(u_profiles, copy(vec(Array(interior(sim.model.velocities.u))[1, 1, :])))
    push!(v_profiles, copy(vec(Array(interior(sim.model.velocities.v))[1, 1, :])))
    push!(T_profiles, copy(vec(Array(interior(sim.model.tracers.T))[1, 1, :])))
    push!(S_profiles, copy(vec(Array(interior(sim.model.tracers.S))[1, 1, :])))
    write_checkpoint()
    println("saved checkpoint at day=", round(sim.model.clock.time/86400,digits=2),
            " | num_saves=", length(saved_times))
end

add_callback!(simulation, save_profiles!, TimeInterval(save_interval))
save_profiles!(simulation)

println("Running $(sim_days)-day v2 ML-forced run with QT at $(site)")
run!(simulation)

if isempty(saved_times) || saved_times[end] != model.clock.time
    save_profiles!(simulation)
end

@save out_file saved_times u_profiles v_profiles T_profiles S_profiles τx τy QT era5_time site_lat site_lon water_depth vert_res Δt0 max_Δt save_interval era5_file model_file glorys_file

println("$(sim_days)-day v2 run completed at $(site)")
println("Saved: ", out_file)
