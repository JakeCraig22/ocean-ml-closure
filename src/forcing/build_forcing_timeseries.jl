module BuildForcingTimeseries

# wraps loading an era5 file and computing the bulk forcing timeseries.
# also exports interp1 which the sims use to interpolate τ/QT at arbitrary
# sim times between era5 hourly samples.

include("load_era5.jl")
include("compute_surface_fluxes.jl")

using .LoadERA5: ERA5PointForcing, load_era5_point
using .ComputeSurfaceFluxes: wind_stress, net_surface_heat_flux

export ForcingSeries, build_forcing_series, interp1

struct ForcingSeries
    time::Vector{Float64}
    τx::Vector{Float64}
    τy::Vector{Float64}
    QT::Vector{Float64}
end

function build_forcing_series(path::String)
    era5 = load_era5_point(path)
    τx, τy = wind_stress(era5.u10, era5.v10)
    QT = net_surface_heat_flux(era5.t2m, era5.sst, era5.u10, era5.v10, era5.ssrd, era5.strd)
    return ForcingSeries(era5.time, τx, τy, QT)
end

# simple linear interp, used inside BC functions so the sim can ask for τ(t) at
# any t. clamp at the ends to avoid out-of-range weirdness.
function interp1(t::Real, ts::AbstractVector, ys::AbstractVector)
    t <= ts[1]   && return ys[1]
    t >= ts[end] && return ys[end]

    i = searchsortedlast(ts, t)
    i == length(ts) && return ys[end]

    t1, t2 = ts[i], ts[i + 1]
    y1, y2 = ys[i], ys[i + 1]
    w = (t - t1) / (t2 - t1)
    return (1 - w) * y1 + w * y2
end

end
