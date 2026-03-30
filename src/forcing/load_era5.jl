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

    names = keys(ds)

    @assert "u10" in names "Missing u10"
    @assert "v10" in names "Missing v10"
    @assert "t2m" in names "Missing t2m"
    @assert "sst" in names "Missing sst"

    time_name = "valid_time" in names ? "valid_time" :
                ("time" in names ? "time" : error("No valid_time/time variable found"))

    raw_time = vec(ds[time_name][:])

    if raw_time[1] isa DateTime
        t0 = raw_time[1]
        time_sec = Float64.(Dates.value.(raw_time .- t0)) ./ 1000.0
    else
        t0 = Float64(raw_time[1])
        time_sec = Float64.(raw_time .- t0)
    end

    u10 = Float64.(coalesce.(vec(ds["u10"][:]), NaN))
    v10 = Float64.(coalesce.(vec(ds["v10"][:]), NaN))
    t2m = Float64.(coalesce.(vec(ds["t2m"][:]), NaN))
    sst = Float64.(coalesce.(vec(ds["sst"][:]), NaN))

    ssrd = "ssrd" in names ? Float64.(coalesce.(vec(ds["ssrd"][:]), NaN)) : zeros(length(time_sec))
    strd = "strd" in names ? Float64.(coalesce.(vec(ds["strd"][:]), NaN)) : zeros(length(time_sec))

    close(ds)

    return ERA5PointForcing(time_sec, u10, v10, t2m, sst, ssrd, strd)
end

end