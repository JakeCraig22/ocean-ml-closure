module ComputeSurfaceFluxes

export wind_speed, wind_stress, net_surface_heat_proxy

function wind_speed(u10::AbstractVector, v10::AbstractVector)
    return sqrt.(u10.^2 .+ v10.^2)
end

function wind_stress(u10::AbstractVector, v10::AbstractVector; ρ_air=1.225, C_d=1.3e-3)
    U = wind_speed(u10, v10)
    τx = ρ_air .* C_d .* U .* u10
    τy = ρ_air .* C_d .* U .* v10
    return τx, τy
end

function net_surface_heat_proxy(t2m::AbstractVector, sst::AbstractVector, ssrd::AbstractVector, strd::AbstractVector;
                                αT=15.0, αsw=1e-6, αlw=1e-6)
    ΔT = t2m .- sst
    Q = αT .* ΔT .+ αsw .* ssrd .+ αlw .* strd
    return Q
end

end