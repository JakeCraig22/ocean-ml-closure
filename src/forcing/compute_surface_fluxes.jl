module ComputeSurfaceFluxes

# bulk formulas for wind stress and net surface heat flux. v1 version — uses
# 80% RH assumption for humidity bc there's no d2m in the old era5 files.
# positive QT = heat INTO ocean (warming). v2 in compute_surface_fluxes_v2.jl

export wind_speed, wind_stress, net_surface_heat_flux

function wind_speed(u10, v10)
    return sqrt.(u10.^2 .+ v10.^2)
end

# standard bulk formula, ρ_air and C_d are the usual textbook values
function wind_stress(u10, v10; ρ_air=1.225, C_d=1.3e-3)
    U = wind_speed(u10, v10)
    τx = ρ_air .* C_d .* U .* u10
    τy = ρ_air .* C_d .* U .* v10
    return τx, τy
end

# tetens formula for saturation humidity, T in K, out kg/kg
function saturation_specific_humidity(T_K; P=101325.0)
    e_sat = @. 611.2 * exp(17.67 * (T_K - 273.15) / (T_K - 29.65))
    return @. 0.622 * e_sat / P
end

# full QT = shortwave + longwave + sensible + latent
function net_surface_heat_flux(t2m, sst, u10, v10, ssrd, strd;
                               albedo  = 0.06,
                               ε       = 0.97,
                               σ_SB    = 5.67e-8,
                               C_H     = 1.1e-3,
                               C_E     = 1.1e-3,
                               ρ_air   = 1.225,
                               c_p_air = 1005.0,
                               L_v     = 2.5e6,
                               RH      = 0.80)

    U = wind_speed(u10, v10)

    # era5 ssrd/strd are J/m^2 accumulated per hour, divide by 3600 to get W/m^2
    Q_sw = @. (1.0 - albedo) * ssrd / 3600.0

    # longwave: down from atmosphere minus Stefan-Boltzmann emission from ocean
    Q_lw = @. ε * strd / 3600.0 - ε * σ_SB * sst^4

    # sensible: warm air → warms ocean
    Q_sh = @. ρ_air * c_p_air * C_H * U * (t2m - sst)

    # latent: evap cools the ocean. sign has to be negative — found this the hard
    # way when mean QT came out +211 W/m^2 instead of the actual ~30.
    q_s = saturation_specific_humidity(sst)
    q_a = RH .* saturation_specific_humidity(t2m)
    Q_lh = @. -ρ_air * L_v * C_E * U * (q_s - q_a)

    return Q_sw .+ Q_lw .+ Q_sh .+ Q_lh
end

end
