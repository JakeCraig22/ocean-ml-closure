using JLD2
using Statistics
using Random
using LinearAlgebra
using Printf

data_path = "data/generated/dataset_many.jld2"

@info "Loading dataset" data_path
d = JLD2.load(data_path)

X_all = d["X_all"]   # (N, Nz, 3)
y_all = d["y_all"]   # (N, Nz)

N, Nz, F = size(X_all)
@assert F == 3

# Train/test split by example
rng = MersenneTwister(42)
idx = collect(1:N)
shuffle!(rng, idx)

ntrain = Int(floor(0.8 * N))
train_idx = idx[1:ntrain]
test_idx  = idx[ntrain+1:end]

@info "Split" ntrain=length(train_idx) ntest=length(test_idx)

function flatten_points(X3::Array{Float64,3}, y2::Array{Float64,2}, ex_idx::Vector{Int})
    M = length(ex_idx) * size(X3, 2)
    Xmat = Array{Float64,2}(undef, M, size(X3, 3))
    yvec = Vector{Float64}(undef, M)
    row = 1
    for i in ex_idx
        for k in 1:size(X3, 2)
            @inbounds Xmat[row, :] .= X3[i, k, :]
            @inbounds yvec[row] = y2[i, k]
            row += 1
        end
    end
    return Xmat, yvec
end

Xtr, ytr = flatten_points(X_all, y_all, train_idx)
Xte, yte = flatten_points(X_all, y_all, test_idx)

# Standardize base features
μ = vec(mean(Xtr, dims=1))
σ = vec(std(Xtr, dims=1))
σ .= ifelse.(σ .== 0.0, 1.0, σ)

Xtrn = (Xtr .- μ') ./ σ'
Xten = (Xte .- μ') ./ σ'

# Build polynomial (degree-2) features from standardized base inputs
function poly2_features(X::Array{Float64,2})
    u    = X[:, 1]
    dudz = X[:, 2]
    dTdz = X[:, 3]

    return hcat(
        u, dudz, dTdz,
        u.^2, dudz.^2, dTdz.^2,
        u .* dudz, u .* dTdz, dudz .* dTdz
    )
end

Φtr = poly2_features(Xtrn)
Φte = poly2_features(Xten)

# Add bias
Φtrb = hcat(ones(size(Φtr, 1)), Φtr)
Φteb = hcat(ones(size(Φte, 1)), Φte)

β = Φtrb \ ytr

ŷtr = Φtrb * β
ŷte = Φteb * β

rmse(a, b) = sqrt(mean((a .- b).^2))

train_rmse = rmse(ŷtr, ytr)
test_rmse  = rmse(ŷte, yte)

@printf("\nPoly2 baseline results:\n")
@printf("  Train RMSE: %.6e\n", train_rmse)
@printf("  Test  RMSE: %.6e\n", test_rmse)

zero_train_rmse = rmse(zeros(length(ytr)), ytr)
zero_test_rmse  = rmse(zeros(length(yte)), yte)

@printf("\nZero baseline:\n")
@printf("  Train RMSE: %.6e\n", zero_train_rmse)
@printf("  Test  RMSE: %.6e\n", zero_test_rmse)

@printf("\nLinear baseline reference (from last run):\n")
@printf("  Test RMSE ≈ 6.002454e-06\n")

out_path = "data/generated/poly2_baseline_model.jld2"
JLD2.@save out_path β μ σ
@info "Saved poly2 model" out_path