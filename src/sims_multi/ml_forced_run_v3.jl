using Oceananigans
using Oceananigans: TimeInterval, IterationInterval
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using JLD2
using Flux
using Statistics

# v3 sim. starts from v2 and tacks on 4 experiments at once:
#   1. κ_T dropped 1e-3 → 1e-4 — less tracer mixing, hopefully holds stratification
#   2. salinity E-P top BC — surface gets saltier under evaporation
#   3. QT bias correction — subtract per-site mean offset between ML and bulk QT
#   4. ENV["USE_BULK"]=1 toggle — replaces ML τ/QT with bulk formula directly.
#      this is the "ceiling experiment" — if the sim with perfect forcing still
#      matches the ML run, then the ML isn't the bottleneck.
# spoiler: v3 was worse than v2 in absolute terms bc the bias correction calibrated
# to bulk (not GLORYS) and they disagree. but v3_ML == v3_BULK at every site, which
# is the paper-worthy finding.

include(joinpath(@__DIR__, "..", "forcing", "load_era5_v2.jl"))
include(joinpath(@__DIR__, "..", "forcing", "compute_surface_fluxes_v2.jl"))
include(joinpath(@__DIR__, "..", "forcing", "build_forcing_timeseries.jl"))

using .LoadERA5v2: load_era5_point_v2
using .ComputeSurfaceFluxesV2: engineered_features, wind_stress, net_surface_heat_flux_v2,
                                wind_speed, saturation_specific_humidity,
                                specific_humidity_from_dewpoint
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

use_bulk = parse(Int, get(ENV, "USE_BULK", "0")) == 1
mode_tag = use_bulk ? "v3bulk" : "v3"

era5_file   = "data/NewEra5/$(site).nc"
model_file  = "data/generated/era5_forcing_results_v2.jld2"
glorys_file = site == "lat30lon-50" ?
    "data/generated/glorys_processed.jld2" :
    "data/generated/glorys_processed_$(site).jld2"

water_depth = 100.0
vert_res    = 16
sim_days = parse(Int, get(ENV, "SIM_DAYS", "30"))

out_file = "output/era5/ml_forced_$(sim_days)day_$(mode_tag)_qt_$(site).jld2"
tmp_file = "output/era5/ml_forced_$(sim_days)day_$(mode_tag)_qt_$(site)_checkpoint.jld2"

Δt0    = 120.0
max_Δt = 1800.0
ρ0     = 1027.0
ρ_fresh = 1000.0
c_p    = 3985.0
L_v    = 2.5e6

# change #1: dropped tracer diffusivity by 10x
ν_eddy = 1e-2
κ_eddy = 1e-4

save_interval = 24 * 3600.0

# --- load era5 ---
data = load_era5_point_v2(era5_file)
u10, v10, t2m, sst, d2m, ssrd, strd, era5_time =
    data.u10, data.v10, data.t2m, data.sst, data.d2m, data.ssrd, data.strd, data.time

Nt = length(era5_time)
stop_time = min(era5_time[end], sim_days * 86400.0)
Nsim = min(Nt, Int(round(stop_time / 3600)) + 1)    # hours within sim window

# --- always compute bulk targets: used for bias correction + USE_BULK=1 path ---
τx_bulk_full, τy_bulk_full = wind_stress(u10, v10)
QT_bulk_full = net_surface_heat_flux_v2(t2m, sst, u10, v10, ssrd, strd, d2m)

# --- latent heat → evaporation rate (for the salinity BC) ---
U = wind_speed(u10, v10)
q_s = saturation_specific_humidity(sst)
q_a = specific_humidity_from_dewpoint(d2m)
ρ_air = 1.225; C_E = 1.1e-3
Q_lh = @. -ρ_air * L_v * C_E * U * (q_s - q_a)   # W/m² (negative during evap)

# E = freshwater leaving the surface in m/s. Q_lh<0 → E>0
E_rate = @. -Q_lh / (L_v * ρ_fresh)

if use_bulk
    # ceiling experiment: feed sim the bulk formula directly, no ML involved
    println("[v3 BULK MODE] site=$site  lat=$site_lat lon=$site_lon  → using bulk formula directly")
    τx = τx_bulk_full
    τy = τy_bulk_full
    QT = QT_bulk_full ./ (ρ0 * c_p)
else
    println("[v3 ML MODE] site=$site  lat=$site_lat lon=$site_lon  → ML + bias correction")
    d = JLD2.load(model_file)
    ml_model = d["model"]
    xμ, xσ, yμ, yσ = d["xμ"], d["xσ"], d["yμ"], d["yσ"]

    eng = engineered_features(u10, v10, t2m, sst, d2m)

    τx_ml = zeros(Float64, Nt)
    τy_ml = zeros(Float64, Nt)
    QT_ml_W = zeros(Float64, Nt)
    for i in 1:Nt
        x = Float64[
            u10[i], v10[i], t2m[i], sst[i], ssrd[i], strd[i], site_lat, site_lon,
            eng[i, 1], eng[i, 2], eng[i, 3], eng[i, 4], eng[i, 5],
        ]
        x_norm = (x .- xμ) ./ xσ
        y_norm = vec(ml_model(Float32.(x_norm)))
        y = y_norm .* yσ .+ yμ
        τx_ml[i] = y[1]
        τy_ml[i] = y[2]
        QT_ml_W[i] = y[3]
    end

    # change #3: bias correction. subtract mean(ML - bulk) over the sim window.
    # idea is to align ML's long-term drift to the bulk formula. in practice this
    # calibrates to the wrong truth (bulk ≠ glorys) which is why v3 ends up worse
    τx_bias = mean(τx_ml[1:Nsim]) - mean(τx_bulk_full[1:Nsim])
    τy_bias = mean(τy_ml[1:Nsim]) - mean(τy_bulk_full[1:Nsim])
    QT_bias = mean(QT_ml_W[1:Nsim]) - mean(QT_bulk_full[1:Nsim])
    println("biases (ML − bulk, sim window): τx=$(round(τx_bias,sigdigits=3))  τy=$(round(τy_bias,sigdigits=3))  QT=$(round(QT_bias,sigdigits=3)) W/m²")

    τx = τx_ml .- τx_bias
    τy = τy_ml .- τy_bias
    QT_W = QT_ml_W .- QT_bias
    QT = QT_W ./ (ρ0 * c_p)
end

println("τx  stats: min=", minimum(τx),  " max=", maximum(τx))
println("τy  stats: min=", minimum(τy),  " max=", maximum(τy))
println("QT  stats (W/m²):  min=", round(minimum(QT) * ρ0 * c_p, sigdigits=4), " max=", round(maximum(QT) * ρ0 * c_p, sigdigits=4))
println("E-P stats (mm/day): min=", round(minimum(E_rate)*86400*1000, sigdigits=4), " max=", round(maximum(E_rate)*86400*1000, sigdigits=4))

# --- glorys init ---
glorys = JLD2.load(glorys_file)
T_init_profile = Float64.(glorys["T_profiles"][1])
S_init_profile = Float64.(glorys["S_profiles"][1])
depth_profile  = Float64.(glorys["target_depth"])
S_surf_ref = S_init_profile[1]    # reference value for the virtual salt flux

# --- grid + BCs ---
grid = RectilinearGrid(size=(2, 2, vert_res), extent=(1.0, 1.0, water_depth),
                      topology=(Periodic, Periodic, Bounded))

# sign-negated for oceananigans' positive-upward top-flux convention
τx_fun(_, _, t) = -interp1(t, era5_time, τx) / ρ0
τy_fun(_, _, t) = -interp1(t, era5_time, τy) / ρ0
QT_fun(_, _, t) = -interp1(t, era5_time, QT)

# change #2: salinity E-P BC.
# virtual salt flux ≈ E * S_surf. positive-into-ocean means S goes UP with evap,
# which is correct (water leaves, salt stays). negate for the upward convention.
S_fun(_, _, t) = -interp1(t, era5_time, E_rate) * S_surf_ref

u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τx_fun), bottom=FluxBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τy_fun), bottom=FluxBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(QT_fun), bottom=FluxBoundaryCondition(0.0))
S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(S_fun),  bottom=FluxBoundaryCondition(0.0))

κ_tracers = (T = κ_eddy, S = κ_eddy)

model = NonhydrostaticModel(
    grid;
    coriolis = FPlane(latitude = site_lat),
    buoyancy = SeawaterBuoyancy(),
    tracers = (:T, :S),
    closure = ScalarDiffusivity(ν = ν_eddy, κ = κ_tracers),
    boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs),
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
    println("progress: day=", round(sim.model.clock.time/86400, digits=2),
            " / ", round(stop_time/86400, digits=2),
            " | iter=", iteration(sim), " | dt=", sim.Δt)
end
add_callback!(simulation, progress, TimeInterval(86400))

mkpath(dirname(out_file))

saved_times = Float64[]
u_profiles  = Vector{Vector{Float64}}()
v_profiles  = Vector{Vector{Float64}}()
T_profiles  = Vector{Vector{Float64}}()
S_profiles  = Vector{Vector{Float64}}()

function write_checkpoint()
    @save tmp_file saved_times u_profiles v_profiles T_profiles S_profiles τx τy QT E_rate era5_time site_lat site_lon water_depth vert_res Δt0 max_Δt save_interval era5_file model_file glorys_file mode_tag
end

function save_profiles!(sim)
    push!(saved_times, sim.model.clock.time)
    push!(u_profiles, copy(vec(Array(interior(sim.model.velocities.u))[1, 1, :])))
    push!(v_profiles, copy(vec(Array(interior(sim.model.velocities.v))[1, 1, :])))
    push!(T_profiles, copy(vec(Array(interior(sim.model.tracers.T))[1, 1, :])))
    push!(S_profiles, copy(vec(Array(interior(sim.model.tracers.S))[1, 1, :])))
    write_checkpoint()
    println("saved checkpoint at day=", round(sim.model.clock.time/86400, digits=2),
            " | num_saves=", length(saved_times))
end

add_callback!(simulation, save_profiles!, TimeInterval(save_interval))
save_profiles!(simulation)

println("Running $(sim_days)-day $(mode_tag) run with QT at $(site)")
run!(simulation)

if isempty(saved_times) || saved_times[end] != model.clock.time
    save_profiles!(simulation)
end

@save out_file saved_times u_profiles v_profiles T_profiles S_profiles τx τy QT E_rate era5_time site_lat site_lon water_depth vert_res Δt0 max_Δt save_interval era5_file model_file glorys_file mode_tag

println("$(sim_days)-day $(mode_tag) run completed at $(site)")
println("Saved: ", out_file)
