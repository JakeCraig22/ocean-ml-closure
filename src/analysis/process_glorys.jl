using NCDatasets
using JLD2
using Statistics

# takes a glorys site folder (data/Copernicus_<site>/) and regrids the 23 native
# depth levels down to 16 levels at 0–100m — same grid the sim uses, so I can
# init the sim directly from this + compare against it later.
#
# usage:
#   julia --project=. src/analysis/process_glorys.jl                 (default lat30lon-50)
#   julia --project=. src/analysis/process_glorys.jl lat-25lon-10    (any site)

site    = isempty(ARGS) ? "lat30lon-50" : ARGS[1]
folder  = "data/Copernicus_$site"
nc_files = filter(f -> endswith(f, ".nc"), readdir(folder; join=true))
isempty(nc_files) && error("No .nc file in $folder")
file = nc_files[1]

# keep the default site unsuffixed bc that's how the v1 sim was originally wired.
# the other sites all use the _suffix form.
out_name = site == "lat30lon-50" ? "glorys_processed.jld2" : "glorys_processed_$site.jld2"
out_path = joinpath("data", "generated", out_name)

println("Processing $file → $out_path")

ds = NCDataset(file)
uo     = ds["uo"]
vo     = ds["vo"]
thetao = ds["thetao"]
so     = ds["so"]

depth = ds["depth"][:]
time  = ds["time"][:]
Nt    = length(time)

# 16 uniform levels matches the sim's vert_res
target_depth = collect(range(0, stop=100, length=16))

u_profiles = Vector{Vector{Float64}}()
v_profiles = Vector{Vector{Float64}}()
T_profiles = Vector{Vector{Float64}}()
S_profiles = Vector{Vector{Float64}}()

# linear interp with edge clamping — nothing fancy
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

# loop each daily snapshot, drop NaN cells (glorys masks out land), regrid
for t in 1:Nt
    u = vec(uo[1, 1, :, t])
    v = vec(vo[1, 1, :, t])
    T = vec(thetao[1, 1, :, t])
    S = vec(so[1, 1, :, t])

    valid = .!isnan.(u) .& .!isnan.(v) .& .!isnan.(T) .& .!isnan.(S)

    d = Float64.(depth[valid])
    u = Float64.(u[valid]); v = Float64.(v[valid])
    T = Float64.(T[valid]); S = Float64.(S[valid])

    push!(u_profiles, interp_profile(d, u, target_depth))
    push!(v_profiles, interp_profile(d, v, target_depth))
    push!(T_profiles, interp_profile(d, T, target_depth))
    push!(S_profiles, interp_profile(d, S, target_depth))
end

close(ds)

mkpath("data/generated")
@save out_path u_profiles v_profiles T_profiles S_profiles target_depth time

println("GLORYS processed successfully ($site)")
println("num timesteps = ", length(u_profiles))
println("profile length = ", length(u_profiles[1]))
println("T init min/max = ", minimum(T_profiles[1]), " / ", maximum(T_profiles[1]))
