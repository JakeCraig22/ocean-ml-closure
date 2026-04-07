using NCDatasets
using JLD2
using Statistics

file = "data/Copernicus_lat30lon-50/cmems_mod_glo_phy_my_0.083deg_P1D-m_1775511387403.nc"

ds = NCDataset(file)

uo = ds["uo"]
vo = ds["vo"]
thetao = ds["thetao"]
so = ds["so"]

depth = ds["depth"][:]
time = ds["time"][:]

Nt = length(time)
target_depth = collect(range(0, stop=100, length=16))

u_profiles = Vector{Vector{Float64}}()
v_profiles = Vector{Vector{Float64}}()
T_profiles = Vector{Vector{Float64}}()
S_profiles = Vector{Vector{Float64}}()

function interp_profile(x_old, y_old, x_new)
    out = Vector{Float64}(undef, length(x_new))
    for (i, x) in enumerate(x_new)
        if x <= x_old[1]
            out[i] = y_old[1]
        elseif x >= x_old[end]
            out[i] = y_old[end]
        else
            j = searchsortedlast(x_old, x)
            x0, x1 = x_old[j], x_old[j + 1]
            y0, y1 = y_old[j], y_old[j + 1]
            out[i] = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        end
    end
    return out
end

for t in 1:Nt
    u = vec(uo[1, 1, :, t])
    v = vec(vo[1, 1, :, t])
    T = vec(thetao[1, 1, :, t])
    S = vec(so[1, 1, :, t])

    valid = .!isnan.(u) .& .!isnan.(v) .& .!isnan.(T) .& .!isnan.(S)

    d = Float64.(depth[valid])
    u = Float64.(u[valid])
    v = Float64.(v[valid])
    T = Float64.(T[valid])
    S = Float64.(S[valid])

    push!(u_profiles, interp_profile(d, u, target_depth))
    push!(v_profiles, interp_profile(d, v, target_depth))
    push!(T_profiles, interp_profile(d, T, target_depth))
    push!(S_profiles, interp_profile(d, S, target_depth))
end

# simple T normalization to roughly match sim scale
T_all = reduce(vcat, T_profiles)
T_min = minimum(T_all)
T_max = maximum(T_all)

for i in eachindex(T_profiles)
    T_profiles[i] = (T_profiles[i] .- T_min) ./ (T_max - T_min)
end

mkpath("data/generated")
@save "data/generated/glorys_processed.jld2" u_profiles v_profiles T_profiles S_profiles target_depth time T_min T_max

println("GLORYS processed successfully")
println("num timesteps = ", length(u_profiles))
println("profile length = ", length(u_profiles[1]))
println("u min/max first = ", minimum(u_profiles[1]), " / ", maximum(u_profiles[1]))
println("T min/max first = ", minimum(T_profiles[1]), " / ", maximum(T_profiles[1]))