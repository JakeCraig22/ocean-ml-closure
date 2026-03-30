using JLD2
using Statistics
using Printf
using CairoMakie

const DEFAULT_RESULTS_PATH = joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_results.jld2")
const DEFAULT_PLOT_DIR = joinpath(@__DIR__, "..", "..", "data", "generated", "plots")

function plot_results(results_path::String=DEFAULT_RESULTS_PATH; out_dir::String=DEFAULT_PLOT_DIR)
    d = JLD2.load(results_path)
    target_names = Vector{String}(d["target_names"])
    rmse_lin = Vector{Float64}(d["rmse_lin"])
    rmse_mlp = Vector{Float64}(d["rmse_mlp"])
    Yte = Array{Float64}(d["Yte"])
    Yhat_lin = Array{Float64}(d["Yhat_lin"])
    Yhat_mlp = Array{Float64}(d["Yhat_mlp"])

    mkpath(out_dir)

    fig1 = Figure(size=(900, 500))
    ax1 = Axis(fig1[1, 1], xlabel="Target", ylabel="RMSE", title="Linear vs MLP RMSE")
    x = 1:length(target_names)
    barplot!(ax1, x .- 0.18, rmse_lin, width=0.32, label="Linear")
    barplot!(ax1, x .+ 0.18, rmse_mlp, width=0.32, label="MLP")
    ax1.xticks = (x, target_names)
    axislegend(ax1, position=:rt)
    save(joinpath(out_dir, "rmse_comparison.png"), fig1)

    for (j, name) in enumerate(target_names)
        fig = Figure(size=(1000, 450))
        ax = Axis(fig[1, 1], xlabel="Test sample", ylabel=name, title="$(name): truth vs predictions")
        n = size(Yte, 1)
        xs = 1:n
        lines!(ax, xs, Yte[:, j], label="Truth")
        lines!(ax, xs, Yhat_lin[:, j], label="Linear")
        lines!(ax, xs, Yhat_mlp[:, j], label="MLP")
        axislegend(ax, position=:rb)
        save(joinpath(out_dir, "$(name)_timeseries.png"), fig)
    end

    println("Saved plots to: $out_dir")
    for j in eachindex(target_names)
        @printf("%-5s | linear RMSE = %.6e | MLP RMSE = %.6e\n", target_names[j], rmse_lin[j], rmse_mlp[j])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_results()
end