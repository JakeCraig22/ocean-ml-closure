using JLD2
using Statistics

include(joinpath(@__DIR__, "load_era5.jl"))
include(joinpath(@__DIR__, "compute_surface_fluxes.jl"))

using .LoadERA5
using .ComputeSurfaceFluxes

function parse_lat_lon_from_filename(path::String)
    fname = lowercase(basename(path))

    m = match(r"lat(-?\d+(?:\.\d+)?)lon(-?\d+(?:\.\d+)?)era5\.nc", fname)
    m === nothing && error("Could not parse lat/lon from filename: $fname")

    lat = parse(Float64, m.captures[1])
    lon = parse(Float64, m.captures[2])
    return lat, lon
end

function build_multisite_dataset(; era5_dir=joinpath(@__DIR__, "..", "..", "data", "ERA5"),
                                   out_path=joinpath(@__DIR__, "..", "..", "data", "generated", "era5_forcing_dataset_multisite.jld2"))

    files = sort(filter(f -> occursin(r"lat-?\d+(?:\.\d+)?lon-?\d+(?:\.\d+)?era5\.nc$", lowercase(basename(f))),
                    readdir(era5_dir; join=true)))
    isempty(files) && error("No .nc files found in $era5_dir")

    X_blocks = Matrix{Float64}[]
    Y_blocks = Matrix{Float64}[]
    source_file = String[]

    println("Found $(length(files)) NetCDF files")
    println()

    for f in files
        lat, lon = parse_lat_lon_from_filename(f)
        data = load_era5_point(f)

        n = length(data.time)

        @assert length(data.u10)  == n
        @assert length(data.v10)  == n
        @assert length(data.t2m)  == n
        @assert length(data.sst)  == n
        @assert length(data.ssrd) == n
        @assert length(data.strd) == n

        τx, τy = wind_stress(data.u10, data.v10)
        QT     = net_surface_heat_proxy(data.t2m, data.sst, data.ssrd, data.strd)

        X = hcat(
            data.u10,
            data.v10,
            data.t2m,
            data.sst,
            data.ssrd,
            data.strd,
            fill(lat, n),
            fill(lon, n),
        )

        Y = hcat(τx, τy, QT)

        mask = vec(all(isfinite, X; dims=2) .& all(isfinite, Y; dims=2))

        X = X[mask, :]
        Y = Y[mask, :]

        push!(X_blocks, X)
        push!(Y_blocks, Y)
        append!(source_file, fill(basename(f), size(X, 1)))

        println("Loaded $(basename(f))")
        println("  lat = $(lat), lon = $(lon)")
        println("  usable samples = $(size(X, 1))")
        println()
    end

    X_all = vcat(X_blocks...)
    Y_all = vcat(Y_blocks...)

    feature_names = ["u10", "v10", "t2m", "sst", "ssrd", "strd", "lat", "lon"]
    target_names  = ["tau_x", "tau_y", "QT"]

    mkpath(dirname(out_path))
    @save out_path X_all Y_all feature_names target_names source_file files

    println("Saved multisite ERA5 forcing dataset to: $out_path")
    println("Total samples : $(size(X_all, 1))")
    println("Features      : $(size(X_all, 2))")
    println("Targets       : $(size(Y_all, 2))")
    println()

    for j in 1:size(X_all, 2)
        col = X_all[:, j]
        println("Feature $(rpad(feature_names[j], 5)) | mean = $(lpad(round(mean(col), sigdigits=8), 12)) | std = $(lpad(round(std(col), sigdigits=8), 12)) | min = $(lpad(round(minimum(col), sigdigits=8), 12)) | max = $(lpad(round(maximum(col), sigdigits=8), 12))")
    end

    println()

    for j in 1:size(Y_all, 2)
        col = Y_all[:, j]
        println("Target  $(rpad(target_names[j], 5)) | mean = $(lpad(round(mean(col), sigdigits=8), 12)) | std = $(lpad(round(std(col), sigdigits=8), 12)) | min = $(lpad(round(minimum(col), sigdigits=8), 12)) | max = $(lpad(round(maximum(col), sigdigits=8), 12))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    build_multisite_dataset()
end