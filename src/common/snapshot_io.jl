module CommonSnapshotIO

using JLD2
using Printf

export save_snapshot_factory

function save_snapshot_factory(outdir::String)
    isdir(outdir) || mkpath(outdir)

    function save_snapshot(sim)
        i = iteration(sim)
        t = time(sim)

        u = Array(interior(sim.model.velocities.u))
        v = Array(interior(sim.model.velocities.v))
        w = Array(interior(sim.model.velocities.w))
        T = Array(interior(sim.model.tracers.T))
        S = Array(interior(sim.model.tracers.S))

        filename = joinpath(outdir, @sprintf("snap_%06d.jld2", i))
        JLD2.@save filename i t u v w T S
        return nothing
    end

    return save_snapshot
end

end