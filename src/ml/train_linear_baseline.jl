
# Train a simple linear regression baseline on dataset_many.jld2.
# Global model: flattens (example, z) points into a single regression.

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

@info "Shapes" N Nz F

# Train/test split by example
rng = MersenneTwister(42)
idx = collect(1:N)
shuffle!(rng, idx)

ntrain = Int(floor(0.8 * N))
train_idx = idx[1:ntrain]
test_idx  = idx[ntrain+1:end]

@info "Split" ntrain=length(train_idx) ntest=length(test_idx)


# Flatten: (example,z) -> rows
function flatten_points(X3::Array{Float64,3}, y2::Array{Float64,2}, ex_idx::Vector{Int})
    # returns Xmat (M, 3), yvec (M)
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

# Standardize features 
μ = vec(mean(Xtr, dims=1))
σ = vec(std(Xtr, dims=1))
σ .= ifelse.(σ .== 0.0, 1.0, σ)

Xtrn = (Xtr .- μ') ./ σ'
Xten = (Xte .- μ') ./ σ'

# Add bias term
Xtrb = hcat(ones(size(Xtrn, 1)), Xtrn)  # (M, 1+F)
Xteb = hcat(ones(size(Xten, 1)), Xten)

# Fit linear regression via least squares
β = Xtrb \ ytr  # (1+F)
@printf("\nCoefficients (standardized feature space):\n")
@printf("  bias  = %.6e\n", β[1])
@printf("  beta_u     = %.6e\n", β[2])
@printf("  beta_dudz  = %.6e\n", β[3])
@printf("  beta_dTdz  = %.6e\n", β[4])

# Predict
ŷtr = Xtrb * β
ŷte = Xteb * β

# Metrics
rmse(a, b) = sqrt(mean((a .- b).^2))

train_rmse = rmse(ŷtr, ytr)
test_rmse  = rmse(ŷte, yte)

@printf("\nLinear baseline results (global regression):\n")
@printf("  Train RMSE: %.6e\n", train_rmse)
@printf("  Test  RMSE: %.6e\n", test_rmse)

# Compare to a dumb baseline: predict 0 residual always
zero_train_rmse = rmse(zeros(length(ytr)), ytr)
zero_test_rmse  = rmse(zeros(length(yte)), yte)

@printf("\nZero baseline (predict residual=0):\n")
@printf("  Train RMSE: %.6e\n", zero_train_rmse)
@printf("  Test  RMSE: %.6e\n", zero_test_rmse)

# Save model params
out_path = "data/generated/linear_baseline_model.jld2"
JLD2.@save out_path β μ σ

@info "Saved linear model" out_path