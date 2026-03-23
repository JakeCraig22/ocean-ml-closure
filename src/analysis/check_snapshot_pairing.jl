# src/analysis/check_snapshot_pairing.jl
using JLD2
using Printf

truth_dir  = "output/truth_surface_stress_tx5e-4_6hr_Nz256"
coarse_dir = "output/surface_stress_tx5e-4_6hr_Nz128"

function snap_list(dir)
    files = sort(filter(f -> startswith(f, "snap_") && endswith(f, ".jld2"), readdir(dir)))
    return files
end

truth_files  = snap_list(truth_dir)
coarse_files = snap_list(coarse_dir)

println("truth  count = ", length(truth_files))
println("coarse count = ", length(coarse_files))

common = intersect(Set(truth_files), Set(coarse_files))
common = sort(collect(common))

println("common count = ", length(common))
if isempty(common)
    error("No matching snapshot filenames between truth and coarse dirs.")
end

# Pick a few samples: first, middle, last
samples = unique([common[1], common[clamp(Int(round(end/2)), 1, length(common))], common[end]])

for f in samples
    tf = joinpath(truth_dir, f)
    cf = joinpath(coarse_dir, f)

    dt = JLD2.load(tf)
    dc = JLD2.load(cf)

    it = dt["i"]; tt = dt["t"]
    ic = dc["i"]; tc = dc["t"]

    Tt = dt["T"]; Tc = dc["T"]

    println("\n--- Snapshot: ", f, " ---")
    @printf(" truth:  i=%d  t=%.2f  T size=%s\n", it, Float64(tt), string(size(Tt)))
    @printf(" coarse: i=%d  t=%.2f  T size=%s\n", ic, Float64(tc), string(size(Tc)))

    # quick sanity: Nz ratio
    Nz_t = size(Tt, 3)
    Nz_c = size(Tc, 3)
    @printf(" Nz_truth=%d Nz_coarse=%d ratio=%.2f\n", Nz_t, Nz_c, Nz_t / Nz_c)
end 