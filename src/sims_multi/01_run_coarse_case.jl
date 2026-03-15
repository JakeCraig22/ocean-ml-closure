using Oceananigans
using Oceananigans: TimeInterval, IterationInterval
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using Printf
using JLD2

# -----------------------------
# forcing argument
# -----------------------------
tx = parse(Float64, ARGS[1])

ρ0 = 1027.0
τx = tx / ρ0

# -----------------------------
# settings
# -----------------------------
water_depth = 100.0
vert_res = 128
Δt0 = 1.0
stop_time = 24 * 60 * 60

# -----------------------------
# grid
# -----------------------------
grid = RectilinearGrid(
    size = (8,8,vert_res),
    extent = (1,1,water_depth),
    topology = (Periodic,Periodic,Bounded)
)

# -----------------------------
# BCs
# -----------------------------
u_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(τx),
    bottom = FluxBoundaryCondition(0.0)
)

v_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(0.0),
    bottom = FluxBoundaryCondition(0.0)
)

# -----------------------------
# model
# -----------------------------
κ_tracers = (T = 1e-5, S = 1e-5)

model = NonhydrostaticModel(
    grid;
    buoyancy = SeawaterBuoyancy(),
    tracers = (:T,:S),
    closure = (AnisotropicMinimumDissipation(),
               ScalarDiffusivity(ν=1e-4, κ=κ_tracers)),
    boundary_conditions = (u=u_bcs, v=v_bcs)
)

set!(model,
    T = (x,y,z) -> -z / water_depth,
    S = (x,y,z) -> 35.0
)

wizard = TimeStepWizard(cfl=0.8, max_Δt=10.0)

simulation = Simulation(model, Δt=Δt0, stop_time=stop_time)

add_callback!(simulation, wizard, IterationInterval(1))

# -----------------------------
# output
# -----------------------------
outdir = "output/multicase/coarse_tx$(tx)_24hr_Nz128"
isdir(outdir) || mkpath(outdir)

function save_snapshot(sim)
    i = iteration(sim)
    t = time(sim)

    u = Array(interior(sim.model.velocities.u))
    w = Array(interior(sim.model.velocities.w))
    T = Array(interior(sim.model.tracers.T))
    S = Array(interior(sim.model.tracers.S))

    filename = joinpath(outdir, @sprintf("snap_%06d.jld2", i))
    @save filename i t u w T S
end

add_callback!(simulation, save_snapshot, TimeInterval(300))

println("Running coarse case tx=", tx)
run!(simulation)