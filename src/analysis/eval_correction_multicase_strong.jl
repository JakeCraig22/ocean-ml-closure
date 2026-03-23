using JLD2
using Statistics
using Printf
using Flux

coarse_dir = "output/multicase/coarse_tx0.002_24hr_Nz128"
truth_dir  = "output/multicase/truth_tx0.002_24hr_Nz256"
model_path = "data/generated/mlp_multicase_model.jld2"

H = 100.0

function horiz_mean_profile(A::AbstractArray{T,3}) where {T}
    Nz = size(A, 3)
    prof = Vector{Float64}(undef, Nz)
    @inbounds for k in 1:Nz
        prof[k] = mean(@view A[:, :, k])
    end
    return prof
end

function central_diff_1d(f::AbstractVector, z::AbstractVector)
    @assert length(f) == length(z)
    Nz = length(z)
    d = Vector{Float64}(undef, Nz)
    d[1]  = (f[2] - f[1]) / (z[2] - z[1])
    d[Nz] = (f[Nz] - f[Nz-1]) / (z[Nz] - z[Nz-1])
    @inbounds for k in 2:(Nz-1)
        d[k] = (f[k+1] - f[k-1]) / (z[k+1] - z[k-1])
    end
    return d
end

function downsample_by_block_mean(f_hi::AbstractVector, Nz_lo::Int)
    Nz_hi = length(f_hi)
    @assert Nz_hi % Nz_lo == 0
    r = Nz_hi ÷ Nz_lo
    f_lo = Vector{Float64}(undef, Nz_lo)
    @inbounds for k in 1:Nz_lo
        i1 = (k-1)*r + 1
        i2 = k*r
        f_lo[k] = mean(@view f_hi[i1:i2])
    end
    return f_lo
end

rmse(a,b) = sqrt(mean((a .- b).^2))

m = JLD2.load(model_path)
mlp_model = m["model"]
μ        = m["Xμ"]
σ        = m["Xσ"]
yμ       = m["yμ"]
yσ       = m["yσ"]

all_files = readdir(coarse_dir)
snap_files = sort(filter(f -> startswith(f, "snap_") && endswith(f, ".jld2"), all_files))
snap_paths = [joinpath(coarse_dir, f) for f in snap_files]

isempty(snap_paths) && error("No snapshots found in $coarse_dir")

rmse_before = Float64[]
rmse_after  = Float64[]
improve_pct = Float64[]
corrs       = Float64[]
names       = String[]

for coarse_path in snap_paths
    snap_name = splitpath(coarse_path)[end]
    truth_path = joinpath(truth_dir, snap_name)
    isfile(truth_path) || continue

    coarse = JLD2.load(coarse_path)
    truth  = JLD2.load(truth_path)

    u_c = coarse["u"]
    T_c = coarse["T"]
    T_t = truth["T"]

    u_prof = horiz_mean_profile(u_c)
    T_prof = horiz_mean_profile(T_c)

    Nz = length(u_prof)
    z = collect(range(-H, 0.0, length=Nz))
    z_norm = z ./ H

    dudz = central_diff_1d(u_prof, z)
    dTdz = central_diff_1d(T_prof, z)

    T_truth_hi = horiz_mean_profile(T_t)
    T_truth_filt = downsample_by_block_mean(T_truth_hi, Nz)

    residual_true = T_truth_filt .- T_prof

    # 4-feature multicase model
    X = hcat(u_prof, dudz, dTdz, z_norm)
    Xn = (X .- μ') ./ σ'
    Xn32 = Float32.(Xn')

    residual_pred = vec(mlp_model(Xn32))
    residual_pred = Float64.(residual_pred .* yσ .+ yμ)

    T_corrected = T_prof .+ residual_pred

    b = rmse(T_prof, T_truth_filt)
    a = rmse(T_corrected, T_truth_filt)

    push!(rmse_before, b)
    push!(rmse_after, a)
    push!(improve_pct, (b - a) / b * 100)

    c = cor(residual_pred, residual_true)
    if !isfinite(c)
        c = 0.0
    end
    push!(corrs, c)

    push!(names, snap_name)
end

N = length(rmse_before)
N == 0 && error("No matched snapshots found.")

@printf("\nEvaluated %d snapshots\n", N)
@printf("Mean RMSE before  = %.6e\n", mean(rmse_before))
@printf("Mean RMSE after   = %.6e\n", mean(rmse_after))
@printf("Mean improvement %% = %.2f %%\n", mean(improve_pct))
@printf("Median improvement %% = %.2f %%\n", median(improve_pct))
@printf("Mean corr(pred,true) = %.4f\n", mean(corrs))

wins = count(rmse_after .< rmse_before)
@printf("Improved snapshots = %d / %d\n", wins, N)

worst_idx = argmin(improve_pct)
best_idx  = argmax(improve_pct)

@printf("\nWorst snapshot:\n")
@printf("  %s | improve=%.2f%% | before=%.6e | after=%.6e | corr=%.3f\n",
        names[worst_idx], improve_pct[worst_idx], rmse_before[worst_idx], rmse_after[worst_idx], corrs[worst_idx])

@printf("\nBest snapshot:\n")
@printf("  %s | improve=%.2f%% | before=%.6e | after=%.6e | corr=%.3f\n",
        names[best_idx], improve_pct[best_idx], rmse_before[best_idx], rmse_after[best_idx], corrs[best_idx])