
# Trains a small MLP on dataset_many.jld2 using Flux (new API).
# IMPORTANT: normalizes BOTH inputs (X) and targets (y).


using JLD2
using Statistics
using Random
using Printf
using Flux

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


# Flatten (example, z) points into a big table
# Flux expects (features, batch)
function flatten_points(X3, y2, ex_idx)
    M = length(ex_idx) * size(X3, 2)
    Xmat = Array{Float32,2}(undef, 3, M)   # (features, batch)
    yvec = Array{Float32,1}(undef, M)
    col = 1
    for i in ex_idx
        for k in 1:size(X3, 2)
            Xmat[:, col] .= Float32.(X3[i, k, :])
            yvec[col] = Float32(y2[i, k])
            col += 1
        end
    end
    return Xmat, yvec
end

Xtr, ytr = flatten_points(X_all, y_all, train_idx)
Xte, yte = flatten_points(X_all, y_all, test_idx)

@printf("Raw target stats (train): max|y|=%.6e  mean|y|=%.6e  std(y)=%.6e\n",
        maximum(abs.(ytr)), mean(abs.(ytr)), std(ytr))

# Normalize inputs using TRAIN stats (per-feature)
Xμ = mean(Xtr, dims=2)
Xσ = std(Xtr, dims=2)
Xσ .= ifelse.(Xσ .== 0f0, 1f0, Xσ)

Xtrn = (Xtr .- Xμ) ./ Xσ
Xten = (Xte .- Xμ) ./ Xσ

# Normalize targets using TRAIN stats
yμ = mean(ytr)
yσ = std(ytr)
yσ = (yσ == 0f0) ? 1f0 : yσ

ytrn = (ytr .- yμ) ./ yσ
yten = (yte .- yμ) ./ yσ

@printf("Normalized target stats (train): mean=%.6e  std=%.6e\n", mean(ytrn), std(ytrn))


model = Chain(
    Dense(3, 32, relu),
    Dense(32, 32, relu),
    Dense(32, 1)
)

loss(m, x, y) = Flux.Losses.mse(vec(m(x)), y)


# Batching
function make_batches(X, y; batchsize=2048, rng=MersenneTwister(42))
    M = size(X, 2)
    order = randperm(rng, M)
    batches = Vector{Tuple{Matrix{Float32}, Vector{Float32}}}()
    j = 1
    while j <= M
        sel = order[j:min(j + batchsize - 1, M)]
        push!(batches, (X[:, sel], y[sel]))
        j += batchsize
    end
    return batches
end

# Train (flux)
opt = Adam(1e-3)
opt_state = Flux.setup(opt, model)

nepoch = 30
for epoch in 1:nepoch
    batches = make_batches(Xtrn, ytrn; batchsize=2048, rng=MersenneTwister(1000 + epoch))
    for (xb, yb) in batches
        grads = gradient(m -> loss(m, xb, yb), model)
        Flux.update!(opt_state, model, grads[1])
    end
    if epoch % 5 == 0
        @printf("epoch %d | train mse (normalized y) %.6e\n", epoch, loss(model, Xtrn, ytrn))
    end
end

# Evaluate
# Report RMSE in original (un-normalized) residual units
pred_te_norm = vec(model(Xten))                  # normalized prediction
pred_te = pred_te_norm .* yσ .+ yμ               # back to original scale

rmse(a, b) = sqrt(mean((Float64.(a) .- Float64.(b)).^2))
test_rmse = rmse(pred_te, yte)

@printf("\nMLP baseline (reported in original residual units):\n")
@printf("  Test RMSE: %.6e\n", test_rmse)

@printf("\nReference baselines (from your previous runs):\n")
@printf("  Poly2 Test RMSE ≈ 4.214193e-06\n")
@printf("  Linear Test RMSE ≈ 6.002454e-06\n")
@printf("  Zero  Test RMSE ≈ 7.343124e-06\n")

# Save model + normalization stats
out_path = "data/generated/mlp_baseline_model.jld2"
JLD2.@save out_path model Xμ Xσ yμ yσ
@info "Saved MLP model" out_path