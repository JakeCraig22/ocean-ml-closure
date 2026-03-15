# src/analysis/eval_rmse_timeseries.jl
#
# RMSE vs time for:
#   - coarse (Nz=128)
#   - MLP ML-closure (Nz=128)
# compared against "truth filtered to coarse" (truth Nz=256 block-averaged to Nz=128)
#
# Output:
#   output/rmse_timeseries.png
#
# Run:
#   julia --project=. src/analysis/eval_rmse_timeseries.jl

using JLD2
using Statistics
using Printf
using CairoMakie

# -----------------------------
# Point these at the runs you actually have
# -----------------------------
truth_dir  = "output/multicase/truth_tx0.002_24hr_Nz256"
coarse_dir = "output/multicase/coarse_tx0.002_24hr_Nz128"
mlp_dir    = "output/multicase/mlclosure_mlp_tx0.002_24hr_Nz128"
# -----------------------------
# Helpers
# -----------------------------
function snap_list(dir::String)
    isdir(dir) || error("Directory not found: $dir")
    files = sort(filter(f -> endswith(f, ".jld2"), readdir(dir)))
    isempty(files) && error("No .jld2 snapshots found in: $dir")
    return files
end

# Truth slice is (Nx, Nz_truth). We need to filter down to (Nx, Nz_coarse).
function filter_truth_to_coarse(Ttruth::AbstractMatrix{<:Real}, Nz_coarse::Int)
    Nx, Nz_truth = size(Ttruth)
    ratio = Nz_truth ÷ Nz_coarse
    if Nz_truth != ratio * Nz_coarse
        error("Truth Nz ($Nz_truth) is not an integer multiple of coarse Nz ($Nz_coarse).")
    end

    Tf = Array{Float64}(undef, Nx, Nz_coarse)
    @inbounds for k in 1:Nz_coarse
        k0 = (k - 1) * ratio + 1
        k1 = k * ratio
        # average over truth vertical block (columns k0:k1) -> one coarse column
        Tf[:, k] .= vec(mean(@view(Ttruth[:, k0:k1]), dims=2))
    end
    return Tf
end

rmse(A::AbstractArray, B::AbstractArray) = sqrt(mean((A .- B) .^ 2))

# -----------------------------
# Load files
# -----------------------------
truth_files  = snap_list(truth_dir)
coarse_files = snap_list(coarse_dir)
mlp_files    = snap_list(mlp_dir)

N = minimum((length(truth_files), length(coarse_files), length(mlp_files)))
@info "Loading snapshot directories..." N=N
@info "Using $N snapshots"

# -----------------------------
# Main loop
# -----------------------------
times      = zeros(Float64, N)
rmse_coarse = zeros(Float64, N)
rmse_mlp    = zeros(Float64, N)

# infer Nz_coarse from first coarse snapshot
c0 = JLD2.load(joinpath(coarse_dir, coarse_files[1]))
T0 = Float64.(c0["T"][:, 1, :])     # (Nx, Nz)
Nz_coarse = size(T0, 2)

@info "Computing RMSE vs time..."

for i in 1:N
    truth  = JLD2.load(joinpath(truth_dir,  truth_files[i]))
    coarse = JLD2.load(joinpath(coarse_dir, coarse_files[i]))
    mlp    = JLD2.load(joinpath(mlp_dir,    mlp_files[i]))

    times[i] = Float64(coarse["t"])

    Ttruth  = Float64.(truth["T"][:, 1, :])     # (Nx, Nz_truth)
    Tcoarse = Float64.(coarse["T"][:, 1, :])    # (Nx, Nz_coarse)
    Tmlp    = Float64.(mlp["T"][:, 1, :])       # (Nx, Nz_coarse)

    Ttruth_f = filter_truth_to_coarse(Ttruth, Nz_coarse)

    rmse_coarse[i] = rmse(Tcoarse, Ttruth_f)
    rmse_mlp[i]    = rmse(Tmlp,    Ttruth_f)
end

@info "Done computing RMSE"

# -----------------------------
# Summary stats
# -----------------------------
mean_coarse = mean(rmse_coarse)
mean_mlp    = mean(rmse_mlp)

med_coarse  = median(rmse_coarse)
med_mlp     = median(rmse_mlp)

final_coarse = rmse_coarse[end]
final_mlp    = rmse_mlp[end]

wins = count(rmse_mlp .< rmse_coarse)
pct  = 100 * wins / N

println()
println("====== RMSE SUMMARY ======")

println("Mean RMSE")
@printf("Coarse = %.16e\n", mean_coarse)
@printf("MLP    = %.16e\n\n", mean_mlp)

println("Median RMSE")
@printf("Coarse = %.16e\n", med_coarse)
@printf("MLP    = %.16e\n\n", med_mlp)

println("Final time RMSE")
@printf("Coarse = %.16e\n", final_coarse)
@printf("MLP    = %.16e\n\n", final_mlp)

@printf("MLP better than coarse on %d / %d snapshots\n", wins, N)
@printf("Percent improved = %.8f%%\n\n", pct)

# -----------------------------
# Plot
# -----------------------------
isdir("output") || mkpath("output")
out_png = joinpath("output", "rmse_timeseries.png")

fig = Figure(size=(1100, 650))
ax  = Axis(fig[1, 1],
           title="Simulation Error vs Truth",
           xlabel="Time (seconds)",
           ylabel="RMSE vs filtered truth")

lines!(ax, times, rmse_coarse, label="Coarse")
lines!(ax, times, rmse_mlp,    label="MLP ML closure")

axislegend(ax, position=:rt)
save(out_png, fig)

println("Saved: ", out_png)