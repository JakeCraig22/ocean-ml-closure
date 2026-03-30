module LoadERA5

using NCDatasets
using Dates

export ERA5PointForcing, load_era5_point

struct ERA5PointForcing
    time::Vector{Float64}   # seconds since start
    u10::Vector{Float64}
    v10::Vector{Float64}
    t2m::Vector{Float64}
    sst::Vector{Float64}
    ssrd::Vector{Float64}
    strd::Vector{Float64}
end

function load_era5_point(path::String)
    ds = NCDataset(path)

    vars = keys(ds.vars)
    @assert "u10" in vars "Missing u10"
    @assert "v10" in vars "Missing v10"
    @assert "t2m" in vars "Missing t2m"
    @assert "sst" in vars "Missing sst"

    time_name = "valid_time" in vars ? "valid_time" : ("time" in vars ? "time" : error("No valid_time/time variable found"))

    raw_time = vec(ds[time_name][:])
    t0 = Float64(raw_time[1])
    time_sec = Float64.(raw_time .- t0) .* 3600.0

    u10 = Float64.(vec(ds["u10"][:]))
    v10 = Float64.(vec(ds["v10"][:]))
    t2m = Float64.(vec(ds["t2m"][:]))
    sst = Float64.(vec(ds["sst"][:]))

    ssrd = "ssrd" in vars ? Float64.(vec(ds["ssrd"][:])) : zeros(length(time_sec))
    strd = "strd" in vars ? Float64.(vec(ds["strd"][:])) : zeros(length(time_sec))

    close(ds)

    return ERA5PointForcing(time_sec, u10, v10, t2m, sst, ssrd, strd)
end

end