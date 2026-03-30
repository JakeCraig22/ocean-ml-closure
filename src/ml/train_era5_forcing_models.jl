using JLD2
using Statistics
using Random
using Printf
using Flux

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_dataset.jld2")
const DEFAULT_RESULTS_PATH = joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_results.jld2")

rmse_vec(yhat::AbstractMatrix, y::AbstractMatrix) = vec(sqrt.(mean((Float64.(yhat) .- Float64.(y)).^2, dims=1)))

function standardize_fit(X::AbstractMatrix)
    μ = vec(mean(X, dims=1))
    σ = vec(std(X, dims=1))
    σ = map(s -> s == 0 ? 1.0 : s, σ)
    return μ, σ
end

standardize_apply(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector) = (X .- permutedims(μ)) ./ permutedims(σ)

function add_bias(X::AbstractMatrix)
    return hcat(ones(size(X,1)), X)
end

function fit_linear_regression(Xtr::AbstractMatrix, Ytr::AbstractMatrix)
    Xb = add_bias(Xtr)
    β = Xb \ Ytr
    return β
end

predict_linear(β, X) = add_bias(X) * β

function make_batches(X::AbstractMatrix, Y::AbstractMatrix; batchsize::Int=64, rng=MersenneTwister(42))
    idx = collect(1:size(X,1))
    shuffle!(rng, idx)
    batches = Tuple{Matrix{Float32}, Matrix{Float32}}[]
    for start in 1:batchsize:length(idx)
        stop = min(start + batchsize - 1, length(idx))
        ids = idx[start:stop]
        xb = permutedims(Float32.(X[ids, :]))
        yb = permutedims(Float32.(Y[ids, :]))
        push!(batches, (xb, yb))
    end
    return batches
end

function train_mlp(Xtr::AbstractMatrix, Ytr::AbstractMatrix; epochs::Int=400, batchsize::Int=64, lr::Float64=1e-3, seed::Int=42)
    F = size(Xtr, 2)
    O = size(Ytr, 2)

    Xμ, Xσ = standardize_fit(Xtr)
    Yμ, Yσ = standardize_fit(Ytr)

    Xtrn = standardize_apply(Xtr, Xμ, Xσ)
    Ytrn = standardize_apply(Ytr, Yμ, Yσ)

    Random.seed!(seed)
    model = Chain(
        Dense(F, 32, relu),
        Dense(32, 32, relu),
        Dense(32, O),
    )

    loss(m, xb, yb) = Flux.Losses.mse(m(xb), yb)

    opt = Adam(lr)
    opt_state = Flux.setup(opt, model)

    for epoch in 1:epochs
        batches = make_batches(Xtrn, Ytrn; batchsize=batchsize, rng=MersenneTwister(seed + epoch))
        for (xb, yb) in batches
            grads = gradient(m -> loss(m, xb, yb), model)
            Flux.update!(opt_state, model, grads[1])
        end
        if epoch % 50 == 0 || epoch == 1
            train_loss = loss(model, permutedims(Float32.(Xtrn)), permutedims(Float32.(Ytrn)))
            @printf("epoch %3d | train mse (normalized targets) = %.6e\n", epoch, train_loss)
        end
    end

    return model, Xμ, Xσ, Yμ, Yσ
end

function predict_mlp(model, X::AbstractMatrix, Xμ, Xσ, Yμ, Yσ)
    Xn = standardize_apply(X, Xμ, Xσ)
    predn = Array(model(permutedims(Float32.(Xn))))'
    pred = predn .* permutedims(Yσ) .+ permutedims(Yμ)
    return pred
end

function train_models(dataset_path::String=DEFAULT_DATASET_PATH; out_path::String=DEFAULT_RESULTS_PATH, seed::Int=42, test_fraction::Float64=0.2)
    d = JLD2.load(dataset_path)
    X = Array{Float64}(d["X"])
    Y = Array{Float64}(d["Y"])
    feature_names = d["feature_names"]
    target_names = d["target_names"]
    time = d["time"]

    N = size(X, 1)
    rng = MersenneTwister(seed)
    idx = collect(1:N)
    shuffle!(rng, idx)

    ntest = max(1, round(Int, test_fraction * N))
    ntrain = N - ntest
    train_idx = idx[1:ntrain]
    test_idx = idx[ntrain+1:end]

    Xtr = X[train_idx, :]
    Ytr = Y[train_idx, :]
    Xte = X[test_idx, :]
    Yte = Y[test_idx, :]

    println("Training split summary")
    @printf("  train samples: %d\n", size(Xtr,1))
    @printf("  test samples : %d\n", size(Xte,1))
    println()

    β = fit_linear_regression(Xtr, Ytr)
    Yhat_lin = predict_linear(β, Xte)
    rmse_lin = rmse_vec(Yhat_lin, Yte)

    println("Training MLP...")
    model, Xμ, Xσ, Yμ, Yσ = train_mlp(Xtr, Ytr; seed=seed)
    Yhat_mlp = predict_mlp(model, Xte, Xμ, Xσ, Yμ, Yσ)
    rmse_mlp = rmse_vec(Yhat_mlp, Yte)

    println()
    println("RMSE by target")
    for j in eachindex(target_names)
        @printf("  %-5s | linear = %12.6e | mlp = %12.6e\n", target_names[j], rmse_lin[j], rmse_mlp[j])
    end

    mkpath(dirname(out_path))
    JLD2.@save out_path X Y feature_names target_names time train_idx test_idx β Xμ Xσ Yμ Yσ rmse_lin rmse_mlp Yhat_lin Yhat_mlp Yte
    println()
    println("Saved results to: $out_path")

    return (; rmse_lin, rmse_mlp, target_names, out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_models()
end