using JLD2
using Plots

qt     = load("output/era5/ml_forced_30day_glorysinit_scaledx005_scaledy010_qt.jld2")
noqt   = load("output/era5/ml_forced_30day_glorysinit_scaledx005_scaledy010.jld2")
glorys = load("data/generated/glorys_processed.jld2")

mkpath("data/generated/plots")

# -----------------------------
# HELPERS
# -----------------------------
function drop_duplicate_times(times, values)
    keep = [1]
    for i in 2:length(times)
        if times[i] != times[i - 1]
            push!(keep, i)
        end
    end
    return times[keep], values[keep]
end

# sim times
times_qt   = qt["saved_times"] ./ 86400
times_noqt = noqt["saved_times"] ./ 86400

# profiles
T_qt   = qt["T_profiles"]
T_noqt = noqt["T_profiles"]
T_real = glorys["T_profiles"]

u_qt   = qt["u_profiles"]
u_noqt = noqt["u_profiles"]
u_real = glorys["u_profiles"]

v_qt   = qt["v_profiles"]
v_noqt = noqt["v_profiles"]
v_real = glorys["v_profiles"]

S_qt   = qt["S_profiles"]
S_noqt = noqt["S_profiles"]
S_real = glorys["S_profiles"]

# use top layer as "surface"
surface_T_qt   = [prof[end] for prof in T_qt]
surface_T_noqt = [prof[end] for prof in T_noqt]
surface_T_real = [prof[end] for prof in T_real]

surface_u_qt   = [prof[end] for prof in u_qt]
surface_u_noqt = [prof[end] for prof in u_noqt]
surface_u_real = [prof[end] for prof in u_real]

surface_v_qt   = [prof[end] for prof in v_qt]
surface_v_noqt = [prof[end] for prof in v_noqt]
surface_v_real = [prof[end] for prof in v_real]

# remove duplicate time-zero saves
times_qt, surface_T_qt   = drop_duplicate_times(times_qt, surface_T_qt)
_,        surface_u_qt   = drop_duplicate_times(qt["saved_times"] ./ 86400, surface_u_qt)
_,        surface_v_qt   = drop_duplicate_times(qt["saved_times"] ./ 86400, surface_v_qt)

times_noqt, surface_T_noqt = drop_duplicate_times(times_noqt, surface_T_noqt)
_,           surface_u_noqt = drop_duplicate_times(noqt["saved_times"] ./ 86400, surface_u_noqt)
_,           surface_v_noqt = drop_duplicate_times(noqt["saved_times"] ./ 86400, surface_v_noqt)

# common lengths for time-series plots
nT = minimum([length(times_qt), length(times_noqt), length(surface_T_real)])
nU = minimum([length(times_qt), length(times_noqt), length(surface_u_real)])
nV = minimum([length(times_qt), length(times_noqt), length(surface_v_real)])

times_T = times_qt[1:nT]
times_U = times_qt[1:nU]
times_V = times_qt[1:nV]

surface_T_qt   = surface_T_qt[1:nT]
surface_T_noqt = surface_T_noqt[1:nT]
surface_T_real = surface_T_real[1:nT]

surface_u_qt   = surface_u_qt[1:nU]
surface_u_noqt = surface_u_noqt[1:nU]
surface_u_real = surface_u_real[1:nU]

surface_v_qt   = surface_v_qt[1:nV]
surface_v_noqt = surface_v_noqt[1:nV]
surface_v_real = surface_v_real[1:nV]

# vertical coordinate for final-day profile plots
vert_res = length(T_qt[end])
z = collect(range(0, stop=100, length=vert_res))  # depth, positive downward

# final-day profiles
T_qt_last   = T_qt[end]
T_noqt_last = T_noqt[end]
T_real_last = T_real[30]

u_qt_last   = u_qt[end]
u_noqt_last = u_noqt[end]
u_real_last = u_real[30]

v_qt_last   = v_qt[end]
v_noqt_last = v_noqt[end]
v_real_last = v_real[30]

S_qt_last   = S_qt[end]
S_noqt_last = S_noqt[end]
S_real_last = S_real[30]

# -----------------------------
# 1) SURFACE TEMPERATURE OVER TIME
# -----------------------------
p1 = plot(times_T, surface_T_noqt,
    label="No QT",
    lw=2)

plot!(p1, times_T, surface_T_qt,
    label="QT (ML)",
    lw=2)

plot!(p1, times_T, surface_T_real,
    label="GLORYS",
    lw=2,
    linestyle=:dash)

xlabel!(p1, "Days")
ylabel!(p1, "Surface Temperature")
title!(p1, "Surface Temperature Evolution (30 Days)")
savefig(p1, "data/generated/plots/surface_temperature_evolution_30day.png")

# -----------------------------
# 2) SURFACE U OVER TIME
# -----------------------------
p2 = plot(times_U, surface_u_noqt,
    label="No QT",
    lw=2)

plot!(p2, times_U, surface_u_qt,
    label="QT (ML)",
    lw=2)

plot!(p2, times_U, surface_u_real,
    label="GLORYS",
    lw=2,
    linestyle=:dash)

xlabel!(p2, "Days")
ylabel!(p2, "Surface U Velocity")
title!(p2, "Surface U Evolution (30 Days)")
savefig(p2, "data/generated/plots/surface_u_evolution_30day.png")

# -----------------------------
# 3) SURFACE V OVER TIME
# -----------------------------
p3 = plot(times_V, surface_v_noqt,
    label="No QT",
    lw=2)

plot!(p3, times_V, surface_v_qt,
    label="QT (ML)",
    lw=2)

plot!(p3, times_V, surface_v_real,
    label="GLORYS",
    lw=2,
    linestyle=:dash)

xlabel!(p3, "Days")
ylabel!(p3, "Surface V Velocity")
title!(p3, "Surface V Evolution (30 Days)")
savefig(p3, "data/generated/plots/surface_v_evolution_30day.png")

# -----------------------------
# 4) FINAL-DAY T PROFILE
# -----------------------------
p4 = plot(T_noqt_last, z,
    label="No QT",
    lw=2)

plot!(p4, T_qt_last, z,
    label="QT (ML)",
    lw=2)

plot!(p4, T_real_last, z,
    label="GLORYS",
    lw=2,
    linestyle=:dash)

xlabel!(p4, "Temperature")
ylabel!(p4, "Depth (m)")
title!(p4, "Final-Day Temperature Profile (Day 30)")
yflip!(p4)
savefig(p4, "data/generated/plots/final_day_T_profile_30day.png")

# -----------------------------
# 5) FINAL-DAY U PROFILE
# -----------------------------
p5 = plot(u_noqt_last, z,
    label="No QT",
    lw=2)

plot!(p5, u_qt_last, z,
    label="QT (ML)",
    lw=2)

plot!(p5, u_real_last, z,
    label="GLORYS",
    lw=2,
    linestyle=:dash)

xlabel!(p5, "U Velocity")
ylabel!(p5, "Depth (m)")
title!(p5, "Final-Day U Profile (Day 30)")
yflip!(p5)
savefig(p5, "data/generated/plots/final_day_u_profile_30day.png")

# -----------------------------
# 6) FINAL-DAY V PROFILE
# -----------------------------
p6 = plot(v_noqt_last, z,
    label="No QT",
    lw=2)

plot!(p6, v_qt_last, z,
    label="QT (ML)",
    lw=2)

plot!(p6, v_real_last, z,
    label="GLORYS",
    lw=2,
    linestyle=:dash)

xlabel!(p6, "V Velocity")
ylabel!(p6, "Depth (m)")
title!(p6, "Final-Day V Profile (Day 30)")
yflip!(p6)
savefig(p6, "data/generated/plots/final_day_v_profile_30day.png")

# -----------------------------
# 7) FINAL-DAY S PROFILE
# -----------------------------
p7 = plot(S_noqt_last, z,
    label="No QT",
    lw=2)

plot!(p7, S_qt_last, z,
    label="QT (ML)",
    lw=2)

plot!(p7, S_real_last, z,
    label="GLORYS",
    lw=2,
    linestyle=:dash)

xlabel!(p7, "Salinity")
ylabel!(p7, "Depth (m)")
title!(p7, "Final-Day Salinity Profile (Day 30)")
yflip!(p7)
savefig(p7, "data/generated/plots/final_day_S_profile_30day.png")

println("Saved plots to: data/generated/plots/")
println(" - surface_temperature_evolution_30day.png")
println(" - surface_u_evolution_30day.png")
println(" - surface_v_evolution_30day.png")
println(" - final_day_T_profile_30day.png")
println(" - final_day_u_profile_30day.png")
println(" - final_day_v_profile_30day.png")
println(" - final_day_S_profile_30day.png")