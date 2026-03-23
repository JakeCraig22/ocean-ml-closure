# src/analysis/eval_correction_many.jl
#
# Evaluate Poly2 correction across MANY snapshot pairs.
# Reports mean RMSE before vs after, improvement stats, and best/worst cases.
#
# Key fixes vs your last version:
# - skip snapshots where "before" RMSE is tiny (percent improvement is meaningless there)
# - report absolute RMSE reduction (robust metric)
# - guard correlation when residual variance is ~0

using JLD2
using Statistics
using Printf

coarse_dir = "output/surface_stress_tx5e-4_6hr_Nz128"
truth_dir  = "output/truth_surface_stress_tx5e-4_6hr_Nz256"
model_path = "data/generated/poly2_baseline_model.jld2"

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

function face_to_center_profile(w_face::AbstractVector)
    return 0.5 .* (w_face[1:end-1] .+ w_face[2:end])
end

rmse(a,b) = sqrt(mean((a .- b).^2))

function poly2_features(X::Array{Float64,2})
    M, F = size(X)
    cols = Vector{Vector{Float64}}()
    sizehint!(cols, F + F + (F*(F-1))÷2)

    # linear terms
    for i in 1:F
        push!(cols, X[:, i])
    end
    # squares
    for i in 1:F
        push!(cols, X[:, i].^2)
    end
    # cross terms
    for i in 1:F
        for j in (i+1):F
            push!(cols, X[:, i] .* X[:, j])
        end
    end

    return hcat(cols...)
end

# Load model
m = JLD2.load(model_path)
β = m["β"]
μ = m["μ"]
σ = m["σ"]

# Snapshot list
all_files = readdir(coarse_dir)
snap_files = sort(filter(f -> startswith(f, "snap_") && endswith(f, ".jld2"), all_files))
snap_paths = [joinpath(coarse_dir, f) for f in snap_files]

isempty(snap_paths) && error("No snapshots found in $coarse_dir")

rmse_before = Float64[]
rmse_after  = Float64[]
improve_pct = Float64[]
corrs       = Float64[]
names       = String[]

# skip threshold: if coarse already matches truth to this level, % improvement is meaningless
min_before_rmse = 1e-6

for coarse_path in snap_paths
    snap_name = splitpath(coarse_path)[end]
    truth_path = joinpath(truth_dir, snap_name)
    isfile(truth_path) || continue

    coarse = JLD2.load(coarse_path)
    truth  = JLD2.load(truth_path)

    u_c = coarse["u"]
    w_c = coarse["w"]
    T_c = coarse["T"]
    T_t = truth["T"]

    u_prof = horiz_mean_profile(u_c)
    T_prof = horiz_mean_profile(T_c)

    Nz = length(u_prof)
    z = collect(range(-H, 0.0, length=Nz))
    z_norm = z ./ H

    w_prof = horiz_mean_profile(w_c)
    if length(w_prof) == Nz + 1
        w_prof = face_to_center_profile(w_prof)
    elseif length(w_prof) != Nz
        continue
    end

    dudz = central_diff_1d(u_prof, z)
    dTdz = central_diff_1d(T_prof, z)

    # Truth filtered to coarse grid
    T_truth_hi = horiz_mean_profile(T_t)
    T_truth_filt = downsample_by_block_mean(T_truth_hi, Nz)

    residual_true = T_truth_filt .- T_prof

    # Predict residual
    X  = hcat(u_prof, w_prof, dudz, dTdz, z_norm)   # (Nz, 5)
    Xn = (X .- μ') ./ σ'
    Φ  = poly2_features(Xn)
    Φb = hcat(ones(size(Φ,1)), Φ)
    residual_pred = Φb * β

    # Apply correction
    T_corrected = T_prof .+ residual_pred

    b = rmse(T_prof, T_truth_filt)
    a = rmse(T_corrected, T_truth_filt)

    # Skip snapshots where the baseline error is tiny
    if b < min_before_rmse
        continue
    end

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
N == 0 && error("After filtering (b < $min_before_rmse), no snapshots remain. Lower min_before_rmse if needed.")

@printf("\nEvaluated %d snapshots (skipped baseline RMSE < %.1e)\n", N, min_before_rmse)
@printf("Mean RMSE before  = %.6e\n", mean(rmse_before))
@printf("Mean RMSE after   = %.6e\n", mean(rmse_after))

abs_red = rmse_before .- rmse_after
@printf("Mean absolute RMSE reduction   = %.6e\n", mean(abs_red))
@printf("Median absolute RMSE reduction = %.6e\n", median(abs_red))

@printf("Mean improvement %%   = %.2f %%\n", mean(improve_pct))
@printf("Median improvement %% = %.2f %%\n", median(improve_pct))
@printf("Mean corr(pred,true) = %.4f\n", mean(corrs))

# Worst / Best by percent improvement
worst_idx = argmin(improve_pct)
best_idx  = argmax(improve_pct)

@printf("\nWorst improvement snapshot:\n")
@printf("  %s | improve=%.2f%% | before=%.6e | after=%.6e | abs_red=%.6e | corr=%.3f\n",
        names[worst_idx], improve_pct[worst_idx], rmse_before[worst_idx], rmse_after[worst_idx], abs_red[worst_idx], corrs[worst_idx])

@printf("\nBest improvement snapshot:\n")
@printf("  %s | improve=%.2f%% | before=%.6e | after=%.6e | abs_red=%.6e | corr=%.3f\n",
        names[best_idx], improve_pct[best_idx], rmse_before[best_idx], rmse_after[best_idx], abs_red[best_idx], corrs[best_idx])