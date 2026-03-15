# src/sims/03_wind_mixing_mlclosure.jl
#
# Coarse Nz=128 sim + Poly2 ML "relaxation" forcing on T:
#   dT/dt += Rhat(z) / tau_relax
#
# Run:
#   julia --project=. src/sims/03_wind_mixing_mlclosure.jl

using Oceananigans
using Oceananigans: TimeInterval, IterationInterval
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using Oceananigans.Forcings: ContinuousForcing
using Statistics
using Printf
using JLD2
using Logging

# -----------------------------
# Make PowerShell piping sane:
# -----------------------------
try
    global_logger(ConsoleLogger(stdout))
    redirect_stderr(stdout)
catch
end

# -----------------------------
# Settings
# -----------------------------
water_depth = 100.0
vert_res    = 128
Δt0         = 1.0
stop_time   = 24 * 60 * 60

# Wind stress
tx = 5e-4
ρ0 = 1027.0
τx = tx / ρ0

# ML relaxation timescale (seconds)
tau_relax = 1800.0

# Clip ML residual magnitude (safety)
R_clip = 2e-4   # same units as model target (your training target)

# Poly2 model file
model_path = "data/generated/poly2_baseline_model.jld2"

# Output directory
const outdir = "output/mlclosure_surface_stress_tx5e-4_6hr_Nz128"
isdir(outdir) || mkpath(outdir)

# -----------------------------
# Helpers
# -----------------------------
function horiz_mean_profile(A::AbstractArray{T,3}) where {T}
    Nz = size(A, 3)
    prof = Vector{Float64}(undef, Nz)
    @inbounds for k in 1:Nz
        prof[k] = mean(@view A[:, :, k])
    end
    return prof
end

function central_diff_1d(f::AbstractVector, z::AbstractVector)
    Nz = length(z)
    d = Vector{Float64}(undef, Nz)
    d[1]  = (f[2] - f[1]) / (z[2] - z[1])
    d[Nz] = (f[Nz] - f[Nz-1]) / (z[Nz] - z[Nz-1])
    @inbounds for k in 2:(Nz-1)
        d[k] = (f[k+1] - f[k-1]) / (z[k+1] - z[k-1])
    end
    return d
end

face_to_center_profile(w_face::AbstractVector) = 0.5 .* (w_face[1:end-1] .+ w_face[2:end])

function poly2_features(X::Array{Float64,2})
    M, F = size(X)
    cols = Vector{Vector{Float64}}()
    sizehint!(cols, F + F + (F*(F-1)) ÷ 2)

    for i in 1:F
        push!(cols, X[:, i])
    end
    for i in 1:F
        push!(cols, X[:, i].^2)
    end
    for i in 1:F
        for j in (i+1):F
            push!(cols, X[:, i] .* X[:, j])
        end
    end
    return hcat(cols...)
end

function stats_str(v::AbstractVector{<:Real})
    return @sprintf("min=% .3e  max=% .3e  mean=% .3e", minimum(v), maximum(v), mean(v))
end

# Map z in [-H, 0] to k in 1:Nz (robust)
@inline function z_to_k(z, H, Nz)
    ξ = (z + H) / H                 # z=-H -> 0, z=0 -> 1
    k = Int(round(ξ * (Nz - 1))) + 1
    return clamp(k, 1, Nz)
end

# -----------------------------
# Load Poly2 model
# -----------------------------
@info "Loading Poly2 model" model_path=model_path
m = JLD2.load(model_path)

# Model file uses greek keys: "β","μ","σ"
beta  = m["β"]
mu    = m["μ"]
sigma = m["σ"]

@info "Poly2 params loaded" beta_len=length(beta) mu_len=length(mu) sigma_len=length(sigma)

# Robust sanity check (prevents silent shape errors later)
if length(mu) != 5 || length(sigma) != 5
    error("Expected mu and sigma to be length 5 (for 5 features). Got mu=$(length(mu)), sigma=$(length(sigma)).")
end

# -----------------------------
# Grid + boundary conditions
# -----------------------------
grid = RectilinearGrid(
    size     = (8, 8, vert_res),
    extent   = (1, 1, water_depth),
    topology = (Periodic, Periodic, Bounded)
)

u_bcs = FieldBoundaryConditions(
    top    = FluxBoundaryCondition(τx),
    bottom = FluxBoundaryCondition(0.0)
)
v_bcs = FieldBoundaryConditions(
    top    = FluxBoundaryCondition(0.0),
    bottom = FluxBoundaryCondition(0.0)
)

# -----------------------------
# ML cache + update callback
# -----------------------------
const H  = water_depth
const Nz = vert_res

# vertical centers in [-H, 0]
zc     = collect(range(-H, 0.0, length=Nz))
z_norm = zc ./ H

# holds predicted residual profile at cell centers
Rhat_cache = Ref(zeros(Float64, Nz))

# one-time debug flags
printed_first_ml      = Ref(false)
printed_forcing_type  = Ref(false)
printed_interior_dims = Ref(false)

function compute_Rhat!(model)
    # Pull interior (no halos)
    uA = Array(interior(model.velocities.u))
    wA = Array(interior(model.velocities.w))
    TA = Array(interior(model.tracers.T))

    if !printed_interior_dims[]
        printed_interior_dims[] = true
        @info "Interior array sizes" u=size(uA) w=size(wA) T=size(TA)
    end

    u_prof = horiz_mean_profile(uA)
    T_prof = horiz_mean_profile(TA)
    w_prof = horiz_mean_profile(wA)

    # w is vertically staggered; might come back as Nz or Nz+1
    if length(w_prof) == Nz + 1
        w_prof = face_to_center_profile(w_prof)
    elseif length(w_prof) != Nz
        @warn "w profile length unexpected; leaving ML cache unchanged" length_w=length(w_prof) Nz=Nz
        return nothing
    end

    dudz = central_diff_1d(u_prof, zc)
    dTdz = central_diff_1d(T_prof, zc)

    # X: Nz x 5
    X = hcat(u_prof, w_prof, dudz, dTdz, z_norm)

    # Standardize (mu, sigma assumed length-5)
    Xn = (X .- mu') ./ sigma'

    # Poly2 expansion + bias
    Φ  = poly2_features(Xn)
    Φb = hcat(ones(size(Φ, 1)), Φ)

    # Predict residual
    Rhat = Φb * beta

    # Clip for safety
    @inbounds for k in 1:Nz
        if Rhat[k] > R_clip
            Rhat[k] = R_clip
        elseif Rhat[k] < -R_clip
            Rhat[k] = -R_clip
        end
    end

    Rhat_cache[] = Rhat

    if !printed_first_ml[]
        printed_first_ml[] = true
        @info "ML cache computed (first time)" w_len=length(w_prof) X_size=size(X) Φ_size=size(Φ) Φb_size=size(Φb)
        @info "Rhat_cache stats (first time)" stats=stats_str(Rhat_cache[])
    end

    return nothing
end

update_ml_cache!(sim) = (compute_Rhat!(sim.model); nothing)

# Forcing for T (reads cache only)
function forcing_T(x, y, z, t)
    k = z_to_k(z, H, Nz)
    return Rhat_cache[][k] / tau_relax
end

ml_T_forcing = ContinuousForcing{Center, Center, Center}(forcing_T)
# -----------------------------
# Model
# -----------------------------
# IMPORTANT FIX: κ must be specified per tracer when tracers = (:T, :S)
κ_tracers = (T = 1e-5, S = 1e-5)

model = NonhydrostaticModel(
    grid;
    buoyancy = SeawaterBuoyancy(),
    tracers  = (:T, :S),
    closure  = (AnisotropicMinimumDissipation(), ScalarDiffusivity(ν=1e-4, κ=κ_tracers)),
    boundary_conditions = (u=u_bcs, v=v_bcs),
    forcing  = (T = ml_T_forcing,)
)

# Initial conditions
set!(model,
    T = (x, y, z) -> -z / water_depth,
    S = (x, y, z) -> 35.0
)

wizard = TimeStepWizard(cfl=0.8, max_Δt=10.0)
simulation = Simulation(model, Δt=Δt0, stop_time=stop_time)

# Standard timestepper callback
add_callback!(simulation, wizard, IterationInterval(1))

# Update ML residual cache once per iteration
add_callback!(simulation, update_ml_cache!, IterationInterval(1))

# Debug: print forcing type once
function print_forcing_type_once(sim)
    if !printed_forcing_type[]
        printed_forcing_type[] = true
        @info "Forcing type debug" forcing_type=typeof(sim.model.forcing.T)
    end
    return nothing
end
add_callback!(simulation, print_forcing_type_once, IterationInterval(1))

# Debug heartbeat
heartbeat(sim) = @info "iter=$(iteration(sim)) time=$(time(sim)) dt=$(sim.Δt) Rhat($(stats_str(Rhat_cache[])))"
add_callback!(simulation, heartbeat, TimeInterval(60))

# Save snapshots
function save_snapshot(sim)
    i = iteration(sim)
    t = time(sim)

    u = Array(interior(sim.model.velocities.u))
    w = Array(interior(sim.model.velocities.w))
    T = Array(interior(sim.model.tracers.T))
    S = Array(interior(sim.model.tracers.S))

    filename = joinpath(outdir, @sprintf("snap_%06d.jld2", i))
    @save filename i t u w T S
    @info "Saved snapshot" filename=filename iter=i time=t
    return nothing
end
add_callback!(simulation, save_snapshot, TimeInterval(300))

@info "Running ML-coupled sim" tau_relax=tau_relax R_clip=R_clip outdir=outdir
run!(simulation)
@info "Done"