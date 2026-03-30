using JLD2
using Statistics
using Printf
using Base.Filesystem: mkpath
using Base: dirname

include(joinpath(@__DIR__, "..", "common", "profiles.jl"))
include(joinpath(@__DIR__, "..", "common", "downsample.jl"))
include(joinpath(@__DIR__, "..", "forcing", "build_forcing_timeseries.jl"))

using .CommonProfiles: horiz_mean_profile, central_diff_1d
using .CommonDownsample: downsample_by_block_mean, snap_list
using .BuildForcingTimeseries: build_forcing_series, interp1

coarse_dir = "output/era5/coarse_Nz128"
truth_dir  = "output/era5/truth_Nz256"
era5_path  = raw"data/ERA5/reanalysis-era5-single-levels-timeseries-sfc6369p5v8.nc"

out_path = "data/generated/dataset_era5.jld2"
H = 100.0

forcing = build_forcing_series(era5_path)

snap_files = snap_list(coarse_dir)

X_list = Vector{Array{Float64,2}}()
y_list = Vector{Vector{Float64}}()
t_list = Float64[]
name_list = String[]
τx_list = Float64[]
τy_list = Float64[]
QT_list = Float64[]

for snap_name in snap_files
    coarse_path = joinpath(coarse_dir, snap_name)
    truth_path  = joinpath(truth_dir, snap_name)

    isfile(truth_path) || continue

    coarse = JLD2.load(coarse_path)
    truth  = JLD2.load(truth_path)

    tnow = Float64(coarse["t"])

    u_c = coarse["u"]
    T_c = coarse["T"]
    T_t = truth["T"]

    u_prof = horiz_mean_profile(u_c)
    T_prof = horiz_mean_profile(T_c)

    Nz = length(u_prof)
    z = collect(range(-H, 0.0, length = Nz))
    z_norm = z ./ H

    dudz = central_diff_1d(u_prof, z)
    dTdz = central_diff_1d(T_prof, z)

    T_truth_hi   = horiz_mean_profile(T_t)
    T_truth_filt = downsample_by_block_mean(T_truth_hi, Nz)

    residual_T = T_truth_filt .- T_prof

    τx_now = interp1(tnow, forcing.time, forcing.τx)
    τy_now = interp1(tnow, forcing.time, forcing.τy)
    QT_now = interp1(tnow, forcing.time, forcing.QT)

    τx_col = fill(τx_now, Nz)
    τy_col = fill(τy_now, Nz)
    QT_col = fill(QT_now, Nz)

    X = hcat(u_prof, dudz, dTdz, z_norm, τx_col, τy_col, QT_col)

    push!(X_list, X)
    push!(y_list, residual_T)
    push!(t_list, tnow)
    push!(name_list, snap_name)
    push!(τx_list, τx_now)
    push!(τy_list, τy_now)
    push!(QT_list, QT_now)
end

N = length(X_list)
N == 0 && error("No examples were processed.")

Nz = size(X_list[1], 1)
F  = size(X_list[1], 2)

X_all = Array{Float64,3}(undef, N, Nz, F)
y_all = Array{Float64,2}(undef, N, Nz)

for i in 1:N
    X_all[i, :, :] .= X_list[i]
    y_all[i, :]    .= y_list[i]
end

@info "ERA5 dataset built" examples = N Nz = Nz features = F
@printf("Residual stats:\n")
@printf("  mean(|y|) = %.6e\n", mean(abs.(y_all)))
@printf("  max(|y|)  = %.6e\n", maximum(abs.(y_all)))

mkpath(dirname(out_path))
JLD2.@save out_path X_all y_all t_list name_list τx_list τy_list QT_list H
@info "Saved ERA5 dataset" out_path = out_path