module LoadERA5v2

# v2 era5 loader. same as v1 except it also pulls d2m (2m dewpoint temp).
# with d2m we get real humidity instead of the 80% RH hack the v1 bulk formula uses.

using NCDatasets
using Dates

export ERA5PointForcingV2, load_era5_point_v2

struct ERA5PointForcingV2
    time::Vector{Float64}
    u10::Vector{Float64}
    v10::Vector{Float64}
    t2m::Vector{Float64}
    sst::Vector{Float64}
    d2m::Vector{Float64}
    ssrd::Vector{Float64}
    strd::Vector{Float64}
end

function load_era5_point_v2(path::String)
    ds = NCDataset(path)
    names = keys(ds)

    # all 7 vars are required for v2
    for v in ["u10","v10","t2m","sst","d2m","ssrd","strd"]
        @assert v in names "Missing $v in $path"
    end

    time_name = "valid_time" in names ? "valid_time" :
                ("time" in names ? "time" : error("No time variable found"))

    raw_time = vec(ds[time_name][:])
    if raw_time[1] isa DateTime
        t0 = raw_time[1]
        time_sec = Float64.(Dates.value.(raw_time .- t0)) ./ 1000.0
    else
        t0 = Float64(raw_time[1])
        time_sec = Float64.(raw_time .- t0)
    end

    coalesced(name) = Float64.(coalesce.(vec(ds[name][:]), NaN))

    u10  = coalesced("u10")
    v10  = coalesced("v10")
    t2m  = coalesced("t2m")
    sst  = coalesced("sst")
    d2m  = coalesced("d2m")
    ssrd = coalesced("ssrd")
    strd = coalesced("strd")

    close(ds)
    return ERA5PointForcingV2(time_sec, u10, v10, t2m, sst, d2m, ssrd, strd)
end

end
