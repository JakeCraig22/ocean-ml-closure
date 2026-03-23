using JLD2
using Statistics
using Printf
using CairoMakie

# -----------------------------
# directories
# -----------------------------
truth_dir  = "output/multicase/truth_tx0.002_24hr_Nz256"
coarse_dir = "output/multicase/coarse_tx0.002_24hr_Nz128"
mlp_dir    = "output/multicase/mlclosure_mlp_tx0.002_24hr_Nz128"

H = 100.0

# -----------------------------
# helpers
# -----------------------------
function snap_list(dir::String)
    isdir(dir) || error("Directory not found: $dir")
    files = sort(filter(f -> endswith(f, ".jld2"), readdir(dir)))
    isempty(files) && error("No .jld2 files found in $dir")
    return files
end

function horiz_mean_profile(A)
    return dropdims(mean(A, dims=(1, 2)), dims=(1, 2))
end

function central_diff(f::AbstractVector, z::AbstractVector)
    Nz = length(z)
    d = zeros(Float64, Nz)

    d[1]   = (f[2] - f[1]) / (z[2] - z[1])
    d[end] = (f[end] - f[end - 1]) / (z[end] - z[end - 1])

    for k in 2:(Nz - 1)
        d[k] = (f[k + 1] - f[k - 1]) / (z[k + 1] - z[k - 1])
    end

    return d
end

function downsample_truth(f_hi::AbstractVector, Nz_lo::Int)
    Nz_hi = length(f_hi)
    Nz_hi % Nz_lo == 0 || error("Truth profile length $Nz_hi is not divisible by coarse length $Nz_lo")

    r = Nz_hi ÷ Nz_lo
    f_lo = zeros(Float64, Nz_lo)

    for k in 1:Nz_lo
        i1 = (k - 1) * r + 1
        i2 = k * r
        f_lo[k] = mean(f_hi[i1:i2])
    end

    return f_lo
end

function compute_profiles(file::String, H::Float64)
    d = JLD2.load(file)

    T = d["T"]
    u = d["u"]

    Tz = horiz_mean_profile(T)
    uz = horiz_mean_profile(u)

    Nz = length(Tz)
    z = collect(range(-H, 0.0, length = Nz))

    dTdz = central_diff(Tz, z)
    dudz = central_diff(uz, z)

    return Tz, uz, dTdz, dudz
end

# -----------------------------
# snapshot lists
# -----------------------------
truth_files  = snap_list(truth_dir)
coarse_files = snap_list(coarse_dir)
mlp_files    = snap_list(mlp_dir)

N = minimum((length(truth_files), length(coarse_files), length(mlp_files)))
println("Using $N snapshots")

# -----------------------------
# storage
# -----------------------------
times = zeros(Float64, N)

mixcontrast_truth  = zeros(Float64, N)
mixcontrast_coarse = zeros(Float64, N)
mixcontrast_mlp    = zeros(Float64, N)

shear_truth  = zeros(Float64, N)
shear_coarse = zeros(Float64, N)
shear_mlp    = zeros(Float64, N)

strat_truth  = zeros(Float64, N)
strat_coarse = zeros(Float64, N)
strat_mlp    = zeros(Float64, N)

# -----------------------------
# main loop
# -----------------------------
for i in 1:N
    coarse = JLD2.load(joinpath(coarse_dir, coarse_files[i]))
    times[i] = Float64(coarse["t"])

    Tz_t, uz_t, dTdz_t, dudz_t = compute_profiles(joinpath(truth_dir, truth_files[i]), H)
    Tz_c, uz_c, dTdz_c, dudz_c = compute_profiles(joinpath(coarse_dir, coarse_files[i]), H)
    Tz_m, uz_m, dTdz_m, dudz_m = compute_profiles(joinpath(mlp_dir, mlp_files[i]), H)

    Nz_c = length(Tz_c)

    # downsample truth to coarse vertical grid
    Tz_tf   = downsample_truth(Tz_t, Nz_c)
    dTdz_tf = downsample_truth(dTdz_t, Nz_c)
    dudz_tf = downsample_truth(dudz_t, Nz_c)

    # ------------------
    # upper-layer temperature contrast
    # smaller magnitude => more homogenized upper ocean
    # ------------------
    top_band   = (Nz_c - 9):Nz_c
    below_band = (Nz_c - 29):(Nz_c - 20)

    mixcontrast_truth[i]  = mean(Tz_tf[top_band]) - mean(Tz_tf[below_band])
    mixcontrast_coarse[i] = mean(Tz_c[top_band])  - mean(Tz_c[below_band])
    mixcontrast_mlp[i]    = mean(Tz_m[top_band])  - mean(Tz_m[below_band])

    # ------------------
    # surface shear metric
    # ------------------
    shear_truth[i]  = mean(abs.(dudz_tf[top_band]))
    shear_coarse[i] = mean(abs.(dudz_c[top_band]))
    shear_mlp[i]    = mean(abs.(dudz_m[top_band]))

    # ------------------
    # surface stratification metric
    # ------------------
    strat_truth[i]  = mean(abs.(dTdz_tf[top_band]))
    strat_coarse[i] = mean(abs.(dTdz_c[top_band]))
    strat_mlp[i]    = mean(abs.(dTdz_m[top_band]))
end

# -----------------------------
# plot
# -----------------------------
isdir("output") || mkpath("output")
out_png = "output/diagnostics_timeseries.png"

fig = Figure(size = (1200, 800))

ax1 = Axis(fig[1, 1], title = "Upper-layer Temperature Contrast", xlabel = "Time (s)", ylabel = "Top - below")
lines!(ax1, times, mixcontrast_truth,  label = "Truth")
lines!(ax1, times, mixcontrast_coarse, label = "Coarse")
lines!(ax1, times, mixcontrast_mlp,    label = "MLP")
axislegend(ax1, position = :rt)

ax2 = Axis(fig[2, 1], title = "Surface Shear |du/dz|", xlabel = "Time (s)", ylabel = "Mean |du/dz|")
lines!(ax2, times, shear_truth,  label = "Truth")
lines!(ax2, times, shear_coarse, label = "Coarse")
lines!(ax2, times, shear_mlp,    label = "MLP")

ax3 = Axis(fig[3, 1], title = "Surface Stratification |dT/dz|", xlabel = "Time (s)", ylabel = "Mean |dT/dz|")
lines!(ax3, times, strat_truth,  label = "Truth")
lines!(ax3, times, strat_coarse, label = "Coarse")
lines!(ax3, times, strat_mlp,    label = "MLP")

save(out_png, fig)

println("Saved: $out_png")

println()
println("Diagnostics summary:")
@printf("  Final upper-layer contrast: Truth=%.6e  Coarse=%.6e  MLP=%.6e\n",
        mixcontrast_truth[end], mixcontrast_coarse[end], mixcontrast_mlp[end])
@printf("  Final surface shear:        Truth=%.6e  Coarse=%.6e  MLP=%.6e\n",
        shear_truth[end], shear_coarse[end], shear_mlp[end])
@printf("  Final surface strat:        Truth=%.6e  Coarse=%.6e  MLP=%.6e\n",
        strat_truth[end], strat_coarse[end], strat_mlp[end])