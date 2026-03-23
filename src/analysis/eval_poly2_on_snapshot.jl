# src/analysis/eval_poly2_on_snapshot.jl
#
# Evaluate the saved Poly2 regression model on ONE snapshot pair.
# Uses features: u(z), w(z), du/dz, dT/dz, z/H
# Predicts residual_T(z) and compares to true residual.

using JLD2
using Statistics
using Printf
using LinearAlgebra

# Paths 
coarse_path = "output/surface_stress_tx5e-4_6hr_Nz128/snap_004330.jld2"
truth_path  = "output/truth_surface_stress_tx5e-4_6hr_Nz256/snap_004330.jld2"

model_path  = "data/generated/poly2_baseline_model.jld2"

H = 100.0

# Helpers 
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
    @assert Nz_hi % Nz_lo == 0 "Nz_hi must be a multiple of Nz_lo"
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

rmse(a, b) = sqrt(mean((a .- b).^2))

# Load model + standardization
m = JLD2.load(model_path)
β = m["β"]
μ = m["μ"]
σ = m["σ"]

# Load snapshots
coarse = JLD2.load(coarse_path)
truth  = JLD2.load(truth_path)

u_c = coarse["u"]
w_c = coarse["w"]
T_c = coarse["T"]
T_t = truth["T"]

# Build features at this snapshot
u_prof = horiz_mean_profile(u_c)
T_prof = horiz_mean_profile(T_c)

Nz = length(u_prof)
z = collect(range(-H, 0.0, length=Nz))
z_norm = z ./ H

w_prof = horiz_mean_profile(w_c)
if length(w_prof) == Nz + 1
    w_prof = face_to_center_profile(w_prof)
elseif length(w_prof) != Nz
    error("Unexpected w_prof length=$(length(w_prof)); expected Nz=$(Nz) or Nz+1=$(Nz+1)")
end

dudz = central_diff_1d(u_prof, z)
dTdz = central_diff_1d(T_prof, z)

X = hcat(u_prof, w_prof, dudz, dTdz, z_norm)  # (Nz, 5)
F = size(X, 2)

@printf("Snapshot features: Nz=%d  F=%d\n", Nz, F)

# Standardize using training μ, σ
Xn = (X .- μ') ./ σ'


# Build true residual
T_truth_hi = horiz_mean_profile(T_t)
T_truth_filt = downsample_by_block_mean(T_truth_hi, Nz)
residual_true = T_truth_filt .- T_prof


# Poly2 features 
function poly2_features(X::Array{Float64,2})
    M, F = size(X)
    cols = Vector{Vector{Float64}}()
    sizehint!(cols, F + F + (F*(F-1))÷2)

    for i in 1:F
        push!(cols, X[:, i])
    end
    for i in 1:F
        push!(cols, X[:, i].^2)
    end
    for i in 1:F
        for j in (i+1):F
            push!(cols, X[:, i] .* X[:, j])
        end
    end

    return hcat(cols...)
end

Φ = poly2_features(Xn)                 # (Nz, nfeat)
Φb = hcat(ones(size(Φ, 1)), Φ)         # add bias

residual_pred = Φb * β                 # (Nz)


# Report
@printf("\nPoly2 snapshot evaluation:\n")
@printf("  RMSE(pred, true) = %.6e\n", rmse(residual_pred, residual_true))
@printf("  max|true|        = %.6e\n", maximum(abs.(residual_true)))
@printf("  max|pred|        = %.6e\n", maximum(abs.(residual_pred)))
@printf("  corr(pred,true)  = %.4f\n", cor(residual_pred, residual_true))

# Optional: show a few values
@printf("\nFirst 5 points (true vs pred):\n")
for k in 1:min(5, Nz)
    @printf("  k=%3d  true=% .6e   pred=% .6e\n", k, residual_true[k], residual_pred[k])
end