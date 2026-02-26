using JLD2
using Statistics

water_depth = 100.0

truth_dir  = "output/truth_surface_stress_tx5e-4_6hr_Nz256"
coarse_dir = "output/surface_stress_tx5e-4_6hr_Nz128"


function last_snapshot(dir)
    files = filter(f -> endswith(f, ".jld2"), readdir(dir, join=true))
    sort!(files)
    return files[end]
end

truth_file  = last_snapshot(truth_dir)
coarse_file = last_snapshot(coarse_dir)

println("Truth snapshot:  ", truth_file)
println("Coarse snapshot: ", coarse_file)

dt = load(truth_file)
dc = load(coarse_file)

T_truth  = dt["T"]
T_coarse = dc["T"]

# Horizontal means

Tz_truth  = dropdims(mean(T_truth,  dims=(1,2)), dims=(1,2))
Tz_coarse = dropdims(mean(T_coarse, dims=(1,2)), dims=(1,2))

# Downsample truth (256 → 128)
ratio = length(Tz_truth) ÷ length(Tz_coarse)

Tz_truth_filtered = [
    mean(Tz_truth[(i-1)*ratio+1 : i*ratio])
    for i in 1:length(Tz_coarse)
]


residual = Tz_truth_filtered .- Tz_coarse

println("\nMax residual magnitude = ", maximum(abs.(residual)))
println("Mean residual magnitude = ", mean(abs.(residual)))

println("\nResidual vector length = ", length(residual))