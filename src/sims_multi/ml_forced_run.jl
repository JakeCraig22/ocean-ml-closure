using Oceananigans
using Oceananigans: TimeInterval, IterationInterval
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using JLD2
using Flux

# v1 ml-forced sim. predicts τx/τy/QT from era5 with v1 mlp, feeds them to a
# 1-d oceananigans column initialized from glorys for the site. glorys is only
# used for init here, the comparison against glorys truth happens in the notebook.
# site via ENV["SITE"], default lat30lon-50. v2 is in ml_forced_run_v2.jl.

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
haskey(SITES, site) || error("Unknown site $site. Choose one of $(collect(keys(SITES)))")
site_lat = SITES[site].lat
site_lon = SITES[site].lon

era5_file   = "data/ERA5/$(site)ERA5.nc"
model_file  = "data/generated/era5_forcing_results_multisite.jld2"
glorys_file = site == "lat30lon-50" ?
    "data/generated/glorys_processed.jld2" :
    "data/generated/glorys_processed_$(site).jld2"

out_file = "output/era5/ml_forced_30day_glorysinit_qt_$(site).jld2"
tmp_file = "output/era5/ml_forced_30day_glorysinit_qt_$(site)_checkpoint.jld2"

water_depth = 100.0
vert_res    = 16       # 16 levels over 100m = ~6m spacing. coarse but enough

sim_days = 30   # flip to 3 for quick smoke tests

Δt0    = 120.0      # starting dt, wizard will adjust
max_Δt = 1800.0     # 30-min max, wizard can't go above this
ρ0     = 1027.0     # reference density for converting τ
c_p    = 3985.0     # ocean heat capacity, for W/m² → K·m/s

# eddy-scale mixing. molecular values (1e-4) made wind momentum pile up at the
# surface and gave unphysical 4 m/s surface jets — see phase 4 memory.
# 1e-2 is what KPP/CATKE would give on average in a boundary layer.
ν_eddy = 1e-2
κ_eddy = 1e-3

save_interval = 24 * 3600.0    # save a snapshot every sim day

# --- load era5 ---
forcing_data = load_era5_point(era5_file)
u10, v10, t2m, sst, ssrd, strd, era5_time =
    forcing_data.u10, forcing_data.v10, forcing_data.t2m, forcing_data.sst,
    forcing_data.ssrd, forcing_data.strd, forcing_data.time

Nt = length(era5_time)
stop_time = min(era5_time[end], sim_days * 86400.0)

# --- load trained mlp ---
d = JLD2.load(model_file)
ml_model = d["model"]
xμ, xσ, yμ, yσ = d["xμ"], d["xσ"], d["yμ"], d["yσ"]

# --- glorys init profiles ---
glorys = JLD2.load(glorys_file)
T_init_profile = Float64.(glorys["T_profiles"][1])   # glorys index 1 = surface
S_init_profile = Float64.(glorys["S_profiles"][1])
depth_profile  = Float64.(glorys["target_depth"])    # 0 = surface, 100 = bottom

println("Loaded GLORYS initial profiles")
println("GLORYS T init min/max = ", minimum(T_init_profile), " / ", maximum(T_init_profile))
println("GLORYS S init min/max = ", minimum(S_init_profile), " / ", maximum(S_init_profile))

# --- run mlp forward for every era5 hour to get τ, QT timeseries ---
τx = zeros(Float64, Nt)
τy = zeros(Float64, Nt)
QT = zeros(Float64, Nt)

for i in 1:Nt
    # 8 inputs in same order the net was trained on
    x = Float64[u10[i], v10[i], t2m[i], sst[i], ssrd[i], strd[i], site_lat, site_lon]

    # z-score → forward → un-z-score
    x_norm = (x .- xμ) ./ xσ
    y_norm = vec(ml_model(Float32.(x_norm)))
    y = y_norm .* yσ .+ yμ

    τx[i] = y[1]
    τy[i] = y[2]
    QT[i] = y[3] / (ρ0 * c_p)   # W/m² → K·m/s
end

println("τx  stats: min=", minimum(τx),  " max=", maximum(τx))
println("τy  stats: min=", minimum(τy),  " max=", maximum(τy))
println("QT  stats (K·m/s): min=", minimum(QT), " max=", maximum(QT))
println("QT  stats (W/m²):  min=", minimum(QT .* ρ0 .* c_p), " max=", maximum(QT .* ρ0 .* c_p))
println("Any NaN τx? ", any(isnan, τx))
println("Any NaN τy? ", any(isnan, τy))
println("Any NaN QT? ", any(isnan, QT))
println("Nt = ", Nt)
println("stop_time = ", stop_time)

# --- grid + BCs ---
grid = RectilinearGrid(size=(2, 2, vert_res), extent=(1.0, 1.0, water_depth),
                      topology=(Periodic, Periodic, Bounded))

# oceananigans top flux is positive-upward (flux OUT of domain). we want
# positive τ/QT = heat/momentum INTO ocean, so negate before passing.
# this was the big phase 4 bug that took forever to find.
τx_fun(_, _, t) = -interp1(t, era5_time, τx) / ρ0
τy_fun(_, _, t) = -interp1(t, era5_time, τy) / ρ0
QT_fun(_, _, t) = -interp1(t, era5_time, QT)

u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τx_fun), bottom=FluxBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τy_fun), bottom=FluxBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(QT_fun), bottom=FluxBoundaryCondition(0.0))

κ_tracers = (T = κ_eddy, S = κ_eddy)

# fplane coriolis at the site latitude. without it wind stress just builds up
# as a zonal jet with no ekman turning — also a phase 4 fix
model = NonhydrostaticModel(
    grid;
    coriolis = FPlane(latitude = site_lat),
    buoyancy = SeawaterBuoyancy(),
    tracers = (:T, :S),
    closure = ScalarDiffusivity(ν = ν_eddy, κ = κ_tracers),
    boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs)
)

# --- initial conditions ---
# glorys depth is positive-down (0=surface, 100=bottom). oceananigans z is
# negative-down (0=surface, -100=bottom). so depth = -z.
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

# --- sim + callbacks ---
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

# --- save state each day into jld2, checkpoint as we go so we don't lose ---
# progress if the run crashes or the computer shuts down
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

    # oceananigans field arrays are indexed [x, y, z], with z=1 at the bottom
    # and z=end at the surface. keep that order here; plots reverse later.
    push!(u_profiles, copy(vec(Array(interior(sim.model.velocities.u))[1, 1, :])))
    push!(v_profiles, copy(vec(Array(interior(sim.model.velocities.v))[1, 1, :])))
    push!(T_profiles, copy(vec(Array(interior(sim.model.tracers.T))[1, 1, :])))
    push!(S_profiles, copy(vec(Array(interior(sim.model.tracers.S))[1, 1, :])))

    write_checkpoint()
    println("saved checkpoint at day=", round(sim.model.clock.time / 86400, digits=2),
            " | num_saves=", length(saved_times))
end

add_callback!(simulation, save_profiles!, TimeInterval(save_interval))
save_profiles!(simulation)      # save day 0 before any stepping happens

println("Running $(sim_days)-day ML-forced run with QT")
run!(simulation)

# belt-and-suspenders: if the last save wasn't exactly at stop_time, grab one more
if isempty(saved_times) || saved_times[end] != model.clock.time
    save_profiles!(simulation)
end

@save out_file saved_times u_profiles v_profiles T_profiles S_profiles τx τy QT era5_time site_lat site_lon water_depth vert_res Δt0 max_Δt save_interval era5_file model_file glorys_file

println("$(sim_days)-day run completed successfully.")
println("Saved final output to: ", out_file)
