using Oceananigans
using Oceananigans: TimeInterval, IterationInterval
using Printf
using JLD2
using Oceananigans.TurbulenceClosures: ScalarDiffusivity

water_depth = 100.0
vert_res = 256
delt_t = 1.0
stop_time = 24 * 60 * 60

grid = RectilinearGrid(
    size = (8, 8, vert_res),
    extent = (1, 1, water_depth),
    topology = (Periodic, Periodic, Bounded)
)

# Stronger wind stress
tx = 2e-3

ρ0 = 1027.0
τx = tx / ρ0

u_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(τx),
    bottom = FluxBoundaryCondition(0.0)
)

v_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(0.0),
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

simulation = Simulation(
    model,
    Δt = delt_t,
    stop_time = stop_time,
)

const outdir = "output/truth_surface_stress_tx2e-3_24hr_Nz256"
isdir(outdir) || mkpath(outdir)

function save_snapshot(sim)
    i = iteration(sim)
    t = time(sim)

    u = Array(sim.model.velocities.u)
    w = Array(sim.model.velocities.w)
    T = Array(sim.model.tracers.T)
    S = Array(sim.model.tracers.S)

    filename = joinpath(outdir, @sprintf("snap_%06d.jld2", i))
    @save filename i t u w T S
    @info "Saved snapshot" filename = filename iter = i time = t
    return nothing
end

heartbeat(sim) = @info "iter=$(iteration(sim)) time=$(time(sim)) Δt=$(sim.Δt)"

add_callback!(simulation, wizard, IterationInterval(1))
add_callback!(simulation, heartbeat, TimeInterval(60))
add_callback!(simulation, save_snapshot, TimeInterval(300))

@info "Running truth wind-driven mixing sim" tx = tx outdir = outdir
run!(simulation)
@info "Simulation complete"