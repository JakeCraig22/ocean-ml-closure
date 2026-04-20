using JLD2
using Flux
using Statistics
using Random
using LinearAlgebra

# v1 trainer. linear reg baseline + small mlp (8→32→32→3). mse loss, adam,
# flat lr=1e-3, 400 epochs. temporal 80/20 split so test data is genuinely
# unseen future hours (random shuffle leaks adjacent-hour samples bc they're
# highly autocorrelated — learned that the hard way in phase 2).

# standard zscore. σ floor at 1e-12 in case some feature is constant
function zscore_fit(X)
    μ = vec(mean(X; dims=1))
    σ = vec(std(X; dims=1))
    σ .= ifelse.(σ .< 1e-12, 1.0, σ)
    return μ, σ
end

zscore_apply(X, μ, σ) = (X .- reshape(μ, 1, :)) ./ reshape(σ, 1, :)

rmse(ytrue, ypred) = sqrt.(vec(mean((ypred .- ytrue).^2; dims=1)))

# analytic least-squares with a bias column. baseline to beat.
function train_linear_regression(Xtr, Ytr, Xte)
    Xtr_aug = hcat(ones(size(Xtr, 1)), Xtr)
    Xte_aug = hcat(ones(size(Xte, 1)), Xte)
    β = Xtr_aug \ Ytr
    return β, Xtr_aug * β, Xte_aug * β
end

function train_mlp(Xtr, Ytr, Xte; hidden=32, epochs=400, lr=1e-3, seed=42)
    Random.seed!(seed)

    in_dim = size(Xtr, 2); out_dim = size(Ytr, 2)

    model = Chain(
        Dense(in_dim, hidden, relu),
        Dense(hidden, hidden, relu),
        Dense(hidden, out_dim),     # identity output bc this is regression not classification
    )

    # flux wants features-as-rows, samples-as-cols → permute
    xtr = Float32.(permutedims(Xtr))
    ytr = Float32.(permutedims(Ytr))
    xte = Float32.(permutedims(Xte))

    opt = Flux.setup(Adam(lr), model)
    loss(m, x, y) = mean((m(x) .- y).^2)

    println("Training MLP...")
    for epoch in 1:epochs
        grads = Flux.gradient(m -> loss(m, xtr, ytr), model)
        Flux.update!(opt, model, grads[1])

        if epoch == 1 || epoch % 50 == 0
            println("epoch $(lpad(epoch, 3)) | train mse (normalized targets) = $(round(loss(model, xtr, ytr), sigdigits=8))")
        end
    end

    return model, Float64.(Array(permutedims(model(xtr)))), Float64.(Array(permutedims(model(xte))))
end

function train_models(;
        dataset_path=joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_dataset_multisite.jld2"),
        out_path=joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_results_multisite.jld2"),
        seed=42)

    @load dataset_path X_all Y_all feature_names target_names source_file

    n = size(X_all, 1)

    # temporal split — first 80% of time is train, last 20% is test
    ntrain    = floor(Int, 0.8 * n)
    train_idx = 1:ntrain
    test_idx  = (ntrain+1):n

    Xtr = X_all[train_idx, :]; Ytr = Y_all[train_idx, :]
    Xte = X_all[test_idx, :];  Yte = Y_all[test_idx, :]

    # fit zscore on train only, apply to both
    xμ, xσ = zscore_fit(Xtr)
    yμ, yσ = zscore_fit(Ytr)
    Xtr_n = zscore_apply(Xtr, xμ, xσ)
    Xte_n = zscore_apply(Xte, xμ, xσ)
    Ytr_n = zscore_apply(Ytr, yμ, yσ)

    println("Training split summary")
    println("  train samples: $(size(Xtr, 1))")
    println("  test samples : $(size(Xte, 1))\n")

    # linear uses unnormalized data, mlp uses normalized — both fine
    _, _, Yte_lin = train_linear_regression(Xtr, Ytr, Xte)
    model, _, Yte_mlp_n = train_mlp(Xtr_n, Ytr_n, Xte_n; seed=seed)

    # unnormalize mlp output so rmse is in real units
    Yte_mlp = Yte_mlp_n .* reshape(yσ, 1, :) .+ reshape(yμ, 1, :)

    rmse_lin = rmse(Yte, Yte_lin)
    rmse_mlp = rmse(Yte, Yte_mlp)

    println("\nRMSE by target")
    for j in 1:length(target_names)
        println("  $(rpad(target_names[j], 5)) | linear = $(round(rmse_lin[j], sigdigits=8)) | mlp = $(round(rmse_mlp[j], sigdigits=8))")
    end

    mkpath(dirname(out_path))
    @save out_path Xtr Xte Ytr Yte Yte_lin Yte_mlp rmse_lin rmse_mlp feature_names target_names train_idx test_idx xμ xσ yμ yσ model

    println("\nSaved results to: $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_models()
end
