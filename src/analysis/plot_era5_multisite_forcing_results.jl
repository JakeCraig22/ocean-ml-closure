using JLD2
using CairoMakie
using Statistics
using Random

function plot_results(; results_path=joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_results_multisite.jld2"),
                        out_dir=joinpath(@__DIR__, "..", "..", "data", "generated", "plots_multisite"),
                        sample_n=3000,
                        seed=42)

    @load results_path Yte Yte_lin Yte_mlp rmse_lin rmse_mlp target_names

    mkpath(out_dir)

    # -----------------------------
    # RMSE bar chart
    # -----------------------------
    fig1 = Figure(size=(900, 500))
    ax1 = Axis(fig1[1, 1],
        title="Multisite RMSE Comparison",
        xlabel="Target",
        ylabel="RMSE"
    )

    x = 1:length(target_names)
    width = 0.35

    barplot!(ax1, x .- width/2, rmse_lin; width=width, label="Linear")
    barplot!(ax1, x .+ width/2, rmse_mlp; width=width, label="MLP")

    ax1.xticks = (x, target_names)
    axislegend(ax1, position=:rt)

    save(joinpath(out_dir, "rmse_comparison_multisite.png"), fig1)

    # -----------------------------
    # Random subset for scatter plots
    # -----------------------------
    n = size(Yte, 1)
    Random.seed!(seed)
    idx = randperm(n)[1:min(sample_n, n)]

    for j in 1:length(target_names)
        truth = Yte[idx, j]
        pred_lin = Yte_lin[idx, j]
        pred_mlp = Yte_mlp[idx, j]

        lo = minimum(vcat(truth, pred_lin, pred_mlp))
        hi = maximum(vcat(truth, pred_lin, pred_mlp))

        fig = Figure(size=(700, 600))
        ax = Axis(fig[1, 1],
            title="$(target_names[j]): truth vs prediction (multisite)",
            xlabel="Truth",
            ylabel="Prediction"
        )

        scatter!(ax, truth, pred_lin; markersize=5, label="Linear")
        scatter!(ax, truth, pred_mlp; markersize=5, label="MLP")
        lines!(ax, [lo, hi], [lo, hi], linestyle=:dash, label="Perfect fit")

        axislegend(ax, position=:rb)

        filename = "$(target_names[j])_scatter_multisite.png"
        save(joinpath(out_dir, filename), fig)
    end

    println("Saved plots to: $out_dir")
    for j in 1:length(target_names)
        println("$(rpad(target_names[j], 5)) | linear RMSE = $(round(rmse_lin[j], sigdigits=8)) | MLP RMSE = $(round(rmse_mlp[j], sigdigits=8))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_results()
end