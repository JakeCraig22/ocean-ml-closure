using JLD2
using Statistics
using Plots

gr()

water_depth = 100.0

function newest_output_run(basedir="output")
    dirs = [joinpath(basedir, d) for d in readdir(basedir) if isdir(joinpath(basedir, d))]
    isempty(dirs) && error("No experiment folders found in output/")
    newest = dirs[argmax(stat(d).mtime for d in dirs)]
    return newest
end

outdir = newest_output_run()
println("Using outdir = ", outdir)

files = filter(f -> endswith(f, ".jld2"), readdir(outdir, join=true))
sort!(files)

if length(files) < 2
    error("Need at least 2 snapshot files in $(outdir) to compare.")
end

first_file = files[1]
last_file  = files[end]

d0 = load(first_file)
d1 = load(last_file)

u0 = d0["u"]; t0 = d0["t"]
u1 = d1["u"]; t1 = d1["t"]

# mean u(z)
uz0 = dropdims(mean(u0, dims=(1,2)), dims=(1,2))
uz1 = dropdims(mean(u1, dims=(1,2)), dims=(1,2))

Nz = length(uz0)
z = range(0, stop=water_depth, length=Nz)

p = plot(uz0, z,
    xlabel="u (m/s)",
    ylabel="Depth (m)",
    title="Mean u profile: first vs last snapshot",
    yflip=true,
    label="first (t=$(round(t0)) s)"
)
plot!(p, uz1, z, label="last (t=$(round(t1)) s)")
display(p)

diff_u = maximum(abs.(uz1 .- uz0))
println("Max |Δu(z)| between first and last = ", diff_u)

println("Press Enter to close...")
readline()