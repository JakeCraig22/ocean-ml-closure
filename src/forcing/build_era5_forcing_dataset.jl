using JLD2
using Statistics
using Printf

include(joinpath(@__DIR__, "load_era5.jl"))
include(joinpath(@__DIR__, "compute_surface_fluxes.jl"))

using .LoadERA5: load_era5_point
using .ComputeSurfaceFluxes: wind_stress, net_surface_heat_proxy

const DEFAULT_ERA5_PATH = joinpath(@__DIR__, "..", "..", "data", "ERA5", "reanalysis-era5-single-levels-timeseries-sfc6369p5v8.nc")
const DEFAULT_OUT_PATH  = joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_dataset.jld2")

function build_dataset(era5_path::String=DEFAULT_ERA5_PATH; out_path::String=DEFAULT_OUT_PATH)
    era5 = load_era5_point(era5_path)

    τx, τy = wind_stress(era5.u10, era5.v10)
    QT = net_surface_heat_proxy(era5.t2m, era5.sst, era5.ssrd, era5.strd)

    feature_names = ["u10", "v10", "t2m", "sst", "ssrd", "strd"]
    target_names  = ["tau_x", "tau_y", "QT"]

    X = Matrix{Float64}(undef, length(era5.time), 6)
    X[:, 1] .= era5.u10
    X[:, 2] .= era5.v10
    X[:, 3] .= era5.t2m
    X[:, 4] .= era5.sst
    X[:, 5] .= era5.ssrd
    X[:, 6] .= era5.strd

    Y = Matrix{Float64}(undef, length(era5.time), 3)
    Y[:, 1] .= τx
    Y[:, 2] .= τy
    Y[:, 3] .= QT

    mkpath(dirname(out_path))
    JLD2.@save out_path X Y feature_names target_names era5_path time=era5.time

    println("Saved ERA5 forcing dataset to: $out_path")
    @printf("Samples: %d | Features: %d | Targets: %d\n", size(X,1), size(X,2), size(Y,2))
    println()

    for j in 1:size(X, 2)
        @printf("Feature %-5s | mean = %12.6f | std = %12.6f | min = %12.6f | max = %12.6f\n",
                feature_names[j], mean(X[:, j]), std(X[:, j]), minimum(X[:, j]), maximum(X[:, j]))
    end
    println()
    for j in 1:size(Y, 2)
        @printf("Target  %-5s | mean = %12.6f | std = %12.6f | min = %12.6f | max = %12.6f\n",
                target_names[j], mean(Y[:, j]), std(Y[:, j]), minimum(Y[:, j]), maximum(Y[:, j]))
    end

    return X, Y, feature_names, target_names, era5.time
end

if abspath(PROGRAM_FILE) == @__FILE__
    build_dataset()
end