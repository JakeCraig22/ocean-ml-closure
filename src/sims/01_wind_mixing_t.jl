using Oceananigans
using Printf
using Dates
using JLD2
using Oceananigans.TurbulenceClosures: ScalarDiffusivity

#-------------
# settings
#-------------
# Numbers that control how big the ocean is
# how detailed is it vertically and how long
# the sim runs

water_depth = 100.0 #depth of water in meters
vert_res = 128 #vertical resolution (num vert layers the water is split into)
delt_t = 1.0 #time step in seconds (starting time step -> seconds per step )
stop_time = 6 *60 * 60 #total time for simulation in seconds (6 hours)

#------------
#grid
#------------
#model divides the water into rectangular boxes
# - 1 box in the east west dir
# - 1 box in the north south dir
# - vert_res boxes in the vertical dir
# representation of one vertical column of the ocean
grid = RectilinearGrid(
    size = (8, 8, vert_res), #num boxes in x,y,z
    extent = (1, 1, water_depth), #size of the box
    topology = (Periodic, Periodic, Bounded)
    # period -> the sides loop around horizontally
    # bounded -> water has a top and bottom
)

#------------
#Surface wind stress
#------------
# Wind pushes on the ocean Surface
# trasfering momentum into the water
# causing mixing

tx = 5e-4 # N/m^2 (strength of wind stress)

ρ0 = 1027.0                 # reference density (kg/m^3)
τx = tx / ρ0                # kinematic stress (m^2/s^2)

u_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(τx),
    bottom = FluxBoundaryCondition(0.0)
)

v_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(0.0),
    bottom = FluxBoundaryCondition(0.0)
)

#------------
#model
#------------
# A fluid model that allows vertical motion
# and turbulence. Temp directly affects density,
# controlling buyoancy and mixing
model = NonhydrostaticModel(
    grid,
    buoyancy = SeawaterBuoyancy(), #Temp controls density
    tracers = (:T, :S), #T to represent temp, S to reperesent Salinity
    closure = (AnisotropicMinimumDissipation(), ScalarDiffusivity(ν=1e-4, κ=1e-5)),
    boundary_conditions = (u = u_bcs, v = v_bcs)
    #used to approximate the small turbulent motions the grid can't see
)

#------------
#initial conditions
#------------

set!(model,
    T = (x, y, z) -> -z / water_depth,
    S = (x, y, z) -> 35.0
     # Temp decreases with depth
)

#------------
#Diagnose
#------------
#automatically adjust such that the simulation
#remains stable

wizard = TimeStepWizard(cfl=0.8, max_Δt=10.0)
# safety factor, and max step allowed

simulation = Simulation(
    model,
    Δt = delt_t, #initial time step
    stop_time = stop_time,
)

#------------
#run
#------------

using Oceananigans: TimeInterval

const outdir = "output/surface_stress_tx5e-4_6hr_Nz128"
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
    @info "Saved snapshot" filename=filename iter=i time=t
    return nothing
end

heartbeat(sim) = @info "iter=$(iteration(sim)) time=$(time(sim)) Δt=$(sim.Δt)"

#attach the TimeStepWizard to the simulation (it updates delta_t every iteration)
add_callback!(simulation, wizard, IterationInterval(1))

#progress logs
add_callback!(simulation, heartbeat, TimeInterval(5))

#save every 5 minutes of model time
add_callback!(simulation, save_snapshot, TimeInterval(300))

@info "Running wind-driven mixing column sim"
run!(simulation)
@info "sim complete"