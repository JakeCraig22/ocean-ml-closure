using JLD2
using Flux
using Statistics
using Random
using LinearAlgebra

# v2 trainer. dataset: era5_forcing_dataset_v2.jld2 (13 inputs with d2m + engineered).
# bigger net 13→64→64→32→3 and 1000 epochs with cosine lr decay 1e-3 → 1e-5.
# temporal 80/20 split, same as v1 so the comparison is fair.
# this combo dropped τ RMSE 3x and QT 1.6x vs v1.

function zscore_fit(X)
    μ = vec(mean(X; dims=1))
    σ = vec(std(X; dims=1))
    σ .= ifelse.(σ .< 1e-12, 1.0, σ)
    return μ, σ
end

zscore_apply(X, μ, σ) = (X .- reshape(μ,1,:)) ./ reshape(σ,1,:)
rmse(y, yhat) = sqrt.(vec(mean((yhat .- y).^2; dims=1)))

function train_linear(Xtr, Ytr, Xte)
    Xtr_aug = hcat(ones(size(Xtr,1)), Xtr)
    Xte_aug = hcat(ones(size(Xte,1)), Xte)
    β = Xtr_aug \ Ytr
    return Xte_aug * β
end

# cosine decay from lr_max → lr_min over `total` epochs. standard schedule,
# gives the net a chance to settle into a decent minimum at the end
function cosine_lr(epoch, total; lr_max=1e-3, lr_min=1e-5)
    return lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π * (epoch-1) / (total-1)))
end

function train_mlp_v2(Xtr, Ytr, Xte; epochs=1000, seed=42, lr_max=1e-3, lr_min=1e-5)
    Random.seed!(seed)

    in_dim  = size(Xtr, 2); out_dim = size(Ytr, 2)

    # bigger net than v1. relu hidden, identity output (regression)
    model = Chain(
        Dense(in_dim, 64, relu),
        Dense(64, 64, relu),
        Dense(64, 32, relu),
        Dense(32, out_dim),
    )

    xtr = Float32.(permutedims(Xtr))
    ytr = Float32.(permutedims(Ytr))
    xte = Float32.(permutedims(Xte))

    opt = Flux.setup(Adam(lr_max), model)
    loss(m, x, y) = mean((m(x) .- y).^2)

    println("Training v2 MLP (13→64→64→32→3, $(epochs) epochs, cosine LR)...")
    for epoch in 1:epochs
        lr = cosine_lr(epoch, epochs; lr_max, lr_min)
        Flux.adjust!(opt, lr)

        grads = Flux.gradient(m -> loss(m, xtr, ytr), model)
        Flux.update!(opt, model, grads[1])

        if epoch == 1 || epoch % 100 == 0 || epoch == epochs
            println("  epoch $(lpad(epoch,4)) | lr=$(round(lr,sigdigits=3)) | train mse=$(round(loss(model, xtr, ytr),sigdigits=6))")
        end
    end

    return model, Float64.(Array(permutedims(model(xte))))
end

function train_v2(;
        dataset_path=joinpath(@__DIR__,"..","..","data","generated","era5_forcing_dataset_v2.jld2"),
        out_path=joinpath(@__DIR__,"..","..","data","generated","era5_forcing_results_v2.jld2"),
        seed=42, epochs=1000)

    @load dataset_path X_all Y_all feature_names target_names source_file

    n = size(X_all, 1)
    ntrain = floor(Int, 0.8 * n)
    train_idx = 1:ntrain
    test_idx  = (ntrain+1):n

    Xtr = X_all[train_idx, :]; Ytr = Y_all[train_idx, :]
    Xte = X_all[test_idx, :];  Yte = Y_all[test_idx, :]

    xμ, xσ = zscore_fit(Xtr); yμ, yσ = zscore_fit(Ytr)
    Xtr_n = zscore_apply(Xtr, xμ, xσ)
    Xte_n = zscore_apply(Xte, xμ, xσ)
    Ytr_n = zscore_apply(Ytr, yμ, yσ)

    println("Split: train=$(size(Xtr,1)) test=$(size(Xte,1)) | features=$(size(Xtr,2))")

    Yte_lin = train_linear(Xtr, Ytr, Xte)
    model, Yte_mlp_n = train_mlp_v2(Xtr_n, Ytr_n, Xte_n; epochs, seed)
    Yte_mlp = Yte_mlp_n .* reshape(yσ,1,:) .+ reshape(yμ,1,:)

    rmse_lin = rmse(Yte, Yte_lin)
    rmse_mlp = rmse(Yte, Yte_mlp)

    println("\nRMSE by target")
    for j in eachindex(target_names)
        println("  $(rpad(target_names[j],5)) | linear=$(round(rmse_lin[j],sigdigits=6)) | mlp=$(round(rmse_mlp[j],sigdigits=6))")
    end

    mkpath(dirname(out_path))
    @save out_path Xtr Xte Ytr Yte Yte_lin Yte_mlp rmse_lin rmse_mlp feature_names target_names train_idx test_idx xμ xσ yμ yσ model
    println("\nSaved: $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_v2()
end
