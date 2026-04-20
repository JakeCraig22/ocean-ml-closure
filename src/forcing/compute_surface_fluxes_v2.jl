module ComputeSurfaceFluxesV2

# v2 bulk formulas. same as v1 but uses d2m from era5 for real humidity instead
# of the 80% RH hack. also exposes engineered_features() which returns the 5
# derived inputs the mlp uses alongside raw era5 vars.

export wind_speed, wind_stress, saturation_specific_humidity,
       specific_humidity_from_dewpoint, net_surface_heat_flux_v2,
       engineered_features

function wind_speed(u10, v10)
    return sqrt.(u10.^2 .+ v10.^2)
end

function wind_stress(u10, v10; ρ_air=1.225, C_d=1.3e-3)
    U = wind_speed(u10, v10)
    τx = ρ_air .* C_d .* U .* u10
    τy = ρ_air .* C_d .* U .* v10
    return τx, τy
end

function saturation_specific_humidity(T_K; P=101325.0)
    e_sat = @. 611.2 * exp(17.67 * (T_K - 273.15) / (T_K - 29.65))
    return @. 0.622 * e_sat / P
end

# dewpoint → actual humidity. standard approx: e = e_sat(T_dew)
function specific_humidity_from_dewpoint(T_d_K; P=101325.0)
    e = @. 611.2 * exp(17.67 * (T_d_K - 273.15) / (T_d_K - 29.65))
    return @. 0.622 * e / P
end

# same layout as v1 QT, but q_air comes from real d2m, not RH*q_sat(t2m)
function net_surface_heat_flux_v2(t2m, sst, u10, v10, ssrd, strd, d2m;
                                  albedo=0.06, ε=0.97, σ_SB=5.67e-8,
                                  C_H=1.1e-3, C_E=1.1e-3,
                                  ρ_air=1.225, c_p_air=1005.0, L_v=2.5e6)
    U = wind_speed(u10, v10)

    Q_sw = @. (1.0 - albedo) * ssrd / 3600.0
    Q_lw = @. ε * strd / 3600.0 - ε * σ_SB * sst^4
    Q_sh = @. ρ_air * c_p_air * C_H * U * (t2m - sst)

    q_s = saturation_specific_humidity(sst)
    q_a = specific_humidity_from_dewpoint(d2m)
    Q_lh = @. -ρ_air * L_v * C_E * U * (q_s - q_a)

    return Q_sw .+ Q_lw .+ Q_sh .+ Q_lh
end

# the 5 features the mlp needs on top of raw era5. these are the nonlinear
# pieces the bulk formula actually uses — giving them to the net directly so
# it doesn't have to rediscover |U|, ΔT, q_sat from scratch. this change alone
# dropped tau RMSE 3x.
function engineered_features(u10, v10, t2m, sst, d2m)
    U    = wind_speed(u10, v10)
    dT   = t2m .- sst
    qsss = saturation_specific_humidity(sst)
    qsa  = saturation_specific_humidity(t2m)
    qa   = specific_humidity_from_dewpoint(d2m)
    return hcat(U, dT, qsss, qsa, qa)
end

end
