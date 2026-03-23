using Oceananigans
using Oceananigans: TimeInterval, IterationInterval
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using Oceananigans.Forcings: ContinuousForcing
using Statistics
using Printf
using JLD2
using Flux
using Logging

try
    global_logger(ConsoleLogger(stdout))
    redirect_stderr(stdout)
catch
end

water_depth = 100.0
vert_res    = 128
Δt0         = 1.0
stop_time   = 24 * 60 * 60

# Stronger wind stress
tx = 2e-3
ρ0 = 1027.0
τx = tx / ρ0

tau_relax = 7200.0
R_clip = 2e-4

ml_enabled = Ref(true)
Rhat_abs_max_allowed  = 8e-5
Rhat_mean_abs_allowed = 2e-6

model_path = "data/generated/mlp_baseline_model.jld2"

const outdir = "output/mlclosure_mlp_surface_stress_tx2e-3_24hr_Nz128"
isdir(outdir) || mkpath(outdir)

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
    @inbounds for k in 2:(Nz - 1)
        d[k] = (f[k + 1] - f[k - 1]) / (z[k + 1] - z[k - 1])
    end
    return d
end

face_to_center_profile(w_face::AbstractVector) = 0.5 .* (w_face[1:end-1] .+ w_face[2:end])

@inline function z_to_k(z, H, Nz)
    ξ = (z + H) / H
    k = Int(round(ξ * (Nz - 1))) + 1
    return clamp(k, 1, Nz)
end

function stats_str(v::AbstractVector{<:Real})
    return @sprintf("min=% .3e  max=% .3e  mean=% .3e", minimum(v), maximum(v), mean(v))
end

@info "Loading MLP model" model_path = model_path
m = JLD2.load(model_path)

mlp_model = m["model"]
mu        = m["Xμ"]
sigma     = m["Xσ"]
y_mu      = m["yμ"]
y_sigma   = m["yσ"]

@info "MLP params loaded"

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

const H  = water_depth
const Nz = vert_res

zc     = collect(range(-H, 0.0, length = Nz))
z_norm = zc ./ H

Rhat_cache = Ref(zeros(Float64, Nz))
printed_first_ml = Ref(false)

function compute_Rhat!(model)
    uA = Array(interior(model.velocities.u))
    wA = Array(interior(model.velocities.w))
    TA = Array(interior(model.tracers.T))

    u_prof = horiz_mean_profile(uA)
    T_prof = horiz_mean_profile(TA)
    w_prof = horiz_mean_profile(wA)

    if length(w_prof) == Nz + 1
        w_prof = face_to_center_profile(w_prof)
    elseif length(w_prof) != Nz
        @warn "w profile length unexpected; leaving ML cache unchanged" length_w = length(w_prof) Nz = Nz
        return nothing
    end

    dudz = central_diff_1d(u_prof, zc)
    dTdz = central_diff_1d(T_prof, zc)

    X = hcat(u_prof, w_prof, dudz, dTdz, z_norm)

    Xn = (X .- mu') ./ sigma'
    Xn32 = Float32.(Xn')

    Rhat = vec(mlp_model(Xn32))
    Rhat = Float64.(Rhat .* y_sigma .+ y_mu)

    absmax  = maximum(abs.(Rhat))
    meanabs = mean(abs.(Rhat))

    if absmax > Rhat_abs_max_allowed || meanabs > Rhat_mean_abs_allowed
        if ml_enabled[]
            @warn "ML forcing disabled by safety gate" absmax = absmax meanabs = meanabs
        end
        ml_enabled[] = false
        Rhat_cache[] = zeros(Float64, Nz)
        return nothing
    end

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
        @info "ML cache first compute" stats = stats_str(Rhat)
    end

    return nothing
end

update_ml_cache!(sim) = (compute_Rhat!(sim.model); nothing)

function forcing_T(x, y, z, t)
    if !ml_enabled[]
        return 0.0
    end
    k = z_to_k(z, H, Nz)
    return Rhat_cache[][k] / tau_relax
end

ml_T_forcing = ContinuousForcing{Center, Center, Center}(forcing_T)

κ_tracers = (T = 1e-5, S = 1e-5)

model = NonhydrostaticModel(
    grid;
    buoyancy = SeawaterBuoyancy(),
    tracers  = (:T, :S),
    closure  = (AnisotropicMinimumDissipation(),
                ScalarDiffusivity(ν = 1e-4, κ = κ_tracers)),
    boundary_conditions = (u = u_bcs, v = v_bcs),
    forcing  = (T = ml_T_forcing,)
)

set!(model,
    T = (x, y, z) -> -z / water_depth,
    S = (x, y, z) -> 35.0
)

wizard = TimeStepWizard(cfl = 0.8, max_Δt = 10.0)
simulation = Simulation(model, Δt = Δt0, stop_time = stop_time)

add_callback!(simulation, wizard, IterationInterval(1))
add_callback!(simulation, update_ml_cache!, IterationInterval(1))

heartbeat(sim) = @info "iter=$(iteration(sim)) time=$(time(sim)) dt=$(sim.Δt) ml_enabled=$(ml_enabled[]) Rhat($(stats_str(Rhat_cache[])))"
add_callback!(simulation, heartbeat, TimeInterval(60))

function save_snapshot(sim)
    i = iteration(sim)
    t = time(sim)

    u = Array(interior(sim.model.velocities.u))
    w = Array(interior(sim.model.velocities.w))
    T = Array(interior(sim.model.tracers.T))
    S = Array(interior(sim.model.tracers.S))

    filename = joinpath(outdir, @sprintf("snap_%06d.jld2", i))
    @save filename i t u w T S
    @info "Saved snapshot" filename = filename iter = i time = t
    return nothing
end
add_callback!(simulation, save_snapshot, TimeInterval(300))

@info "Running MLP closure sim" outdir = outdir tx = tx tau_relax = tau_relax R_clip = R_clip gate_absmax = Rhat_abs_max_allowed gate_meanabs = Rhat_mean_abs_allowed
run!(simulation)
@info "Done"