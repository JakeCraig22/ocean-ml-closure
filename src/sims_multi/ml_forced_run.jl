using Oceananigans
using Oceananigans: TimeInterval, IterationInterval
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using JLD2
using Flux

include(joinpath(@__DIR__, "..", "forcing", "load_era5.jl"))
include(joinpath(@__DIR__, "..", "forcing", "build_forcing_timeseries.jl"))

using .LoadERA5: load_era5_point
using .BuildForcingTimeseries: interp1

# -----------------------------
# SETTINGS
# -----------------------------
era5_file   = raw"data/ERA5/lat30lon-50ERA5.nc"
model_file  = raw"data/generated/era5_forcing_results_multisite.jld2"
glorys_file = raw"data/generated/glorys_processed.jld2"

out_file = raw"output/era5/ml_forced_30day_glorysinit_scaledx005_scaledy010_qt.jld2"
tmp_file = raw"output/era5/ml_forced_30day_glorysinit_scaledx005_scaledy010_qt_checkpoint.jld2"

site_lat = 30.0
site_lon = -50.0

water_depth = 100.0
vert_res    = 16

Δt0    = 120.0
max_Δt = 1800.0
ρ0     = 1027.0

save_interval   = 24 * 3600.0
forcing_scale_x = 0.05
forcing_scale_y = 0.10
forcing_scale_QT = 1e-6

# -----------------------------
# LOAD ERA5
# -----------------------------
forcing_data = load_era5_point(era5_file)

u10       = forcing_data.u10
v10       = forcing_data.v10
t2m       = forcing_data.t2m
sst       = forcing_data.sst
ssrd      = forcing_data.ssrd
strd      = forcing_data.strd
era5_time = forcing_data.time

Nt = length(era5_time)
stop_time = min(era5_time[end], 30 * 86400.0)

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
d = JLD2.load(model_file)
ml_model = d["model"]
xμ = d["xμ"]
xσ = d["xσ"]
yμ = d["yμ"]
yσ = d["yσ"]

# -----------------------------
# LOAD GLORYS INITIAL PROFILES
# -----------------------------
glorys = JLD2.load(glorys_file)

T_init_profile = Float64.(glorys["T_profiles"][1])
S_init_profile = Float64.(glorys["S_profiles"][1])
depth_profile  = Float64.(glorys["target_depth"])

println("Loaded GLORYS initial profiles")
println("GLORYS T init min/max = ", minimum(T_init_profile), " / ", maximum(T_init_profile))
println("GLORYS S init min/max = ", minimum(S_init_profile), " / ", maximum(S_init_profile))

# -----------------------------
# PREDICT ML FORCING SERIES
# -----------------------------
τx = zeros(Float64, Nt)
τy = zeros(Float64, Nt)
QT = zeros(Float64, Nt)

for i in 1:Nt
    x = Float64[
        u10[i], v10[i], t2m[i], sst[i], ssrd[i], strd[i], site_lat, site_lon
    ]

    x_norm = (x .- xμ) ./ xσ
    y_norm = vec(ml_model(Float32.(x_norm)))
    y = y_norm .* yσ .+ yμ

    τx[i] = forcing_scale_x * y[1]
    τy[i] = forcing_scale_y * y[2]
    QT[i] = -forcing_scale_QT * y[3]   
end

println("τx stats: min=", minimum(τx), " max=", maximum(τx))
println("τy stats: min=", minimum(τy), " max=", maximum(τy))
println("QT stats: min=", minimum(QT), " max=", maximum(QT))
println("Any NaN τx? ", any(isnan, τx))
println("Any NaN τy? ", any(isnan, τy))
println("Any NaN QT? ", any(isnan, QT))
println("Nt = ", Nt)
println("RAW y[3] stats: min=", minimum(QT ./ forcing_scale_QT), 
        " max=", maximum(QT ./ forcing_scale_QT))
println("stop_time = ", stop_time)

# -----------------------------
# GRID
# -----------------------------
grid = RectilinearGrid(
    size = (2, 2, vert_res),
    extent = (1.0, 1.0, water_depth),
    topology = (Periodic, Periodic, Bounded)
)

# -----------------------------
# BOUNDARY CONDITIONS
# -----------------------------
τx_fun(x, y, t) = interp1(t, era5_time, τx) / ρ0
τy_fun(x, y, t) = interp1(t, era5_time, τy) / ρ0
QT_fun(x, y, t) = interp1(t, era5_time, QT)

u_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(τx_fun),
    bottom = FluxBoundaryCondition(0.0)
)

v_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(τy_fun),
    bottom = FluxBoundaryCondition(0.0)
)

T_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(QT_fun),
    bottom = FluxBoundaryCondition(0.0)
)

κ_tracers = (T = 1e-5, S = 1e-5)

# -----------------------------
# MODEL
# -----------------------------
model = NonhydrostaticModel(
    grid;
    buoyancy = SeawaterBuoyancy(),
    tracers = (:T, :S),
    closure = ScalarDiffusivity(ν = 1e-4, κ = κ_tracers),
    boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs)
)

# -----------------------------
# INITIAL CONDITIONS
# -----------------------------
function interp_profile(depth_val, depth_grid, profile)
    if depth_val <= depth_grid[1]
        return profile[1]
    elseif depth_val >= depth_grid[end]
        return profile[end]
    end

    j = searchsortedlast(depth_grid, depth_val)
    j = min(j, length(depth_grid) - 1)

    d0, d1 = depth_grid[j], depth_grid[j + 1]
    p0, p1 = profile[j], profile[j + 1]

    return p0 + (p1 - p0) * (depth_val - d0) / (d1 - d0)
end

T_init(x, y, z) = interp_profile(-z, depth_profile, T_init_profile)
S_init(x, y, z) = interp_profile(-z, depth_profile, S_init_profile)

set!(model, T = T_init, S = S_init)

# -----------------------------
# SIMULATION
# -----------------------------
simulation = Simulation(model, Δt = Δt0, stop_time = stop_time)

wizard = TimeStepWizard(cfl = 0.8, max_Δt = max_Δt)
add_callback!(simulation, wizard, IterationInterval(1))

function progress(sim)
    t_days = sim.model.clock.time / 86400
    total_days = stop_time / 86400
    println("progress: day=", round(t_days, digits=2),
            " / ", round(total_days, digits=2),
            " | iter=", iteration(sim),
            " | dt=", sim.Δt)
end

add_callback!(simulation, progress, TimeInterval(86400))

# -----------------------------
# SAVE
# -----------------------------
mkpath(dirname(out_file))

saved_times = Float64[]
u_profiles  = Vector{Vector{Float64}}()
v_profiles  = Vector{Vector{Float64}}()
T_profiles  = Vector{Vector{Float64}}()
S_profiles  = Vector{Vector{Float64}}()

function write_checkpoint()
    @save tmp_file saved_times u_profiles v_profiles T_profiles S_profiles τx τy QT era5_time site_lat site_lon water_depth vert_res Δt0 max_Δt save_interval forcing_scale_x forcing_scale_y forcing_scale_QT era5_file model_file glorys_file
end

function save_profiles!(sim)
    push!(saved_times, sim.model.clock.time)

    u_arr = Array(interior(sim.model.velocities.u))
    v_arr = Array(interior(sim.model.velocities.v))
    T_arr = Array(interior(sim.model.tracers.T))
    S_arr = Array(interior(sim.model.tracers.S))

    push!(u_profiles, copy(vec(u_arr[1, 1, :])))
    push!(v_profiles, copy(vec(v_arr[1, 1, :])))
    push!(T_profiles, copy(vec(T_arr[1, 1, :])))
    push!(S_profiles, copy(vec(S_arr[1, 1, :])))

    write_checkpoint()

    println("saved checkpoint at day=",
            round(sim.model.clock.time / 86400, digits=2),
            " | num_saves=", length(saved_times))
end

add_callback!(simulation, save_profiles!, TimeInterval(save_interval))
save_profiles!(simulation)

println("Running 1-day ML-forced run with QT")
run!(simulation)

if isempty(saved_times) || saved_times[end] != model.clock.time
    save_profiles!(simulation)
end

@save out_file saved_times u_profiles v_profiles T_profiles S_profiles τx τy QT era5_time site_lat site_lon water_depth vert_res Δt0 max_Δt save_interval forcing_scale_x forcing_scale_y forcing_scale_QT era5_file model_file glorys_file

println("1-day run completed successfully.")
println("Saved final output to: ", out_file)
