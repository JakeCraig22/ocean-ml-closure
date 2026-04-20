using JLD2
using Statistics

# v2 dataset builder. reads data/NewEra5/ (the files with d2m) and adds 5
# engineered features on top of the 8 raw ones → 13 features total.
# output: data/generated/era5_forcing_dataset_v2.jld2

include(joinpath(@__DIR__, "load_era5_v2.jl"))
include(joinpath(@__DIR__, "compute_surface_fluxes_v2.jl"))

using .LoadERA5v2
using .ComputeSurfaceFluxesV2

function parse_lat_lon_from_filename(path::String)
    fname = lowercase(basename(path))
    m = match(r"lat(-?\d+(?:\.\d+)?)lon(-?\d+(?:\.\d+)?)\.nc", fname)
    m === nothing && error("Could not parse lat/lon from filename: $fname")
    return parse(Float64, m.captures[1]), parse(Float64, m.captures[2])
end

function build_multisite_dataset_v2(;
        era5_dir=joinpath(@__DIR__, "..", "..", "data", "NewEra5"),
        out_path=joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_dataset_v2.jld2"))

    files = sort(filter(f -> endswith(lowercase(f), ".nc"), readdir(era5_dir; join=true)))
    isempty(files) && error("No .nc files in $era5_dir")

    X_blocks = Matrix{Float64}[]
    Y_blocks = Matrix{Float64}[]
    source_file = String[]

    println("Found $(length(files)) NetCDF files in NewEra5")

    for f in files
        lat, lon = parse_lat_lon_from_filename(f)
        data = load_era5_point_v2(f)
        n = length(data.time)

        # targets: bulk formula with real d2m humidity
        τx, τy = wind_stress(data.u10, data.v10)
        QT     = net_surface_heat_flux_v2(data.t2m, data.sst, data.u10, data.v10,
                                          data.ssrd, data.strd, data.d2m)

        # 5 engineered features (|U|, ΔT, q_sat(sst), q_sat(t2m), q_air)
        eng = engineered_features(data.u10, data.v10, data.t2m, data.sst, data.d2m)

        # 8 raw + 5 engineered = 13 inputs
        X = hcat(data.u10, data.v10, data.t2m, data.sst, data.ssrd, data.strd,
                 fill(lat, n), fill(lon, n), eng)
        Y = hcat(τx, τy, QT)

        mask = vec(all(isfinite, X; dims=2) .& all(isfinite, Y; dims=2))
        X = X[mask, :]; Y = Y[mask, :]

        push!(X_blocks, X); push!(Y_blocks, Y)
        append!(source_file, fill(basename(f), size(X, 1)))

        println("  $(basename(f))  lat=$lat lon=$lon  samples=$(size(X,1))")
    end

    X_all = vcat(X_blocks...)
    Y_all = vcat(Y_blocks...)

    feature_names = ["u10","v10","t2m","sst","ssrd","strd","lat","lon",
                     "U","dT","q_sat_sst","q_sat_t2m","q_air"]
    target_names = ["tau_x","tau_y","QT"]

    mkpath(dirname(out_path))
    @save out_path X_all Y_all feature_names target_names source_file files

    println("\nSaved: $out_path")
    println("Total samples: $(size(X_all,1))  |  Features: $(size(X_all,2))  |  Targets: $(size(Y_all,2))")
    for j in axes(X_all, 2)
        col = X_all[:, j]
        println("  $(rpad(feature_names[j],10)) mean=$(round(mean(col),sigdigits=5))  std=$(round(std(col),sigdigits=5))")
    end
    for j in axes(Y_all, 2)
        col = Y_all[:, j]
        println("  $(rpad(target_names[j],10)) mean=$(round(mean(col),sigdigits=5))  std=$(round(std(col),sigdigits=5))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    build_multisite_dataset_v2()
end
