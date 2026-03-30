using JLD2
using Flux
using Statistics
using Random
using LinearAlgebra

function zscore_fit(X::Matrix{Float64})
    μ = vec(mean(X; dims=1))
    σ = vec(std(X; dims=1))
    σ .= ifelse.(σ .< 1e-12, 1.0, σ)
    return μ, σ
end

function zscore_apply(X::Matrix{Float64}, μ::Vector{Float64}, σ::Vector{Float64})
    return (X .- reshape(μ, 1, :)) ./ reshape(σ, 1, :)
end

function rmse(ytrue::Matrix{Float64}, ypred::Matrix{Float64})
    return sqrt.(vec(mean((ypred .- ytrue).^2; dims=1)))
end

function train_linear_regression(Xtr::Matrix{Float64}, Ytr::Matrix{Float64}, Xte::Matrix{Float64})
    Xtr_aug = hcat(ones(size(Xtr, 1)), Xtr)
    Xte_aug = hcat(ones(size(Xte, 1)), Xte)

    β = Xtr_aug \ Ytr
    Ytr_hat = Xtr_aug * β
    Yte_hat = Xte_aug * β

    return β, Ytr_hat, Yte_hat
end

function train_mlp(Xtr::Matrix{Float64}, Ytr::Matrix{Float64}, Xte::Matrix{Float64};
                   hidden=32, epochs=400, lr=1e-3, seed=42)

    Random.seed!(seed)

    in_dim = size(Xtr, 2)
    out_dim = size(Ytr, 2)

    model = Chain(
        Dense(in_dim, hidden, relu),
        Dense(hidden, hidden, relu),
        Dense(hidden, out_dim),
    )

    xtr = Float32.(permutedims(Xtr))
    ytr = Float32.(permutedims(Ytr))
    xte = Float32.(permutedims(Xte))

    opt = Flux.setup(Adam(lr), model)

    loss(m, x, y) = mean((m(x) .- y).^2)

    println("Training MLP...")
    for epoch in 1:epochs
        grads = Flux.gradient(model) do m
            loss(m, xtr, ytr)
        end
        Flux.update!(opt, model, grads[1])

        if epoch == 1 || epoch % 50 == 0
            l = loss(model, xtr, ytr)
            println("epoch $(lpad(epoch, 3)) | train mse (normalized targets) = $(round(l, sigdigits=8))")
        end
    end

    ytr_hat = Array(permutedims(model(xtr)))
    yte_hat = Array(permutedims(model(xte)))

    return model, Float64.(ytr_hat), Float64.(yte_hat)
end

function train_models(; dataset_path=joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_dataset_multisite.jld2"),
                        out_path=joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_results_multisite.jld2"),
                        seed=42)

    @load dataset_path X_all Y_all feature_names target_names source_file

    n = size(X_all, 1)
    Random.seed!(seed)

    idx = collect(1:n)
    shuffle!(idx)

    ntrain = floor(Int, 0.8 * n)
    train_idx = idx[1:ntrain]
    test_idx  = idx[ntrain+1:end]

    Xtr = X_all[train_idx, :]
    Ytr = Y_all[train_idx, :]
    Xte = X_all[test_idx, :]
    Yte = Y_all[test_idx, :]

    xμ, xσ = zscore_fit(Xtr)
    yμ, yσ = zscore_fit(Ytr)

    Xtr_n = zscore_apply(Xtr, xμ, xσ)
    Xte_n = zscore_apply(Xte, xμ, xσ)
    Ytr_n = zscore_apply(Ytr, yμ, yσ)

    println("Training split summary")
    println("  train samples: $(size(Xtr, 1))")
    println("  test samples : $(size(Xte, 1))")
    println()

    β, Ytr_lin, Yte_lin = train_linear_regression(Xtr, Ytr, Xte)

    model, Ytr_mlp_n, Yte_mlp_n = train_mlp(Xtr_n, Ytr_n, Xte_n)

    Ytr_mlp = Ytr_mlp_n .* reshape(yσ, 1, :) .+ reshape(yμ, 1, :)
    Yte_mlp = Yte_mlp_n .* reshape(yσ, 1, :) .+ reshape(yμ, 1, :)

    rmse_lin = rmse(Yte, Yte_lin)
    rmse_mlp = rmse(Yte, Yte_mlp)

    println()
    println("RMSE by target")
    for j in 1:length(target_names)
        println("  $(rpad(target_names[j], 5)) | linear = $(round(rmse_lin[j], sigdigits=8)) | mlp = $(round(rmse_mlp[j], sigdigits=8))")
    end

    mkpath(dirname(out_path))
    @save out_path Xtr Xte Ytr Yte Yte_lin Yte_mlp rmse_lin rmse_mlp feature_names target_names train_idx test_idx xμ xσ yμ yσ

    println()
    println("Saved results to: $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_models()
end