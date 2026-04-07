using JLD2
using Statistics

sim = load( "output/era5/ml_forced_5day_glorysinit_scaledx005_scaledy010_qt.jld2")
real = load("data/generated/glorys_processed.jld2")

println("sim saved_times = ", sim["saved_times"])

u_sim = sim["u_profiles"][end]
v_sim = sim["v_profiles"][end]
T_sim = sim["T_profiles"][end]
S_sim = sim["S_profiles"][end]

u_real = real["u_profiles"][5]
v_real = real["v_profiles"][5]
T_real = real["T_profiles"][5]
S_real = real["S_profiles"][5]

function rmse(a, b)
    sqrt(mean((a .- b).^2))
end

println("---- DAY 5 CHECK ----")
println("u RMSE = ", rmse(u_sim, u_real))
println("v RMSE = ", rmse(v_sim, v_real))
println("T RMSE = ", rmse(T_sim, T_real))
println("S RMSE = ", rmse(S_sim, S_real))

println("\nSIM vs REAL ranges:")
println("u sim min/max = ", minimum(u_sim), " / ", maximum(u_sim))
println("u real min/max = ", minimum(u_real), " / ", maximum(u_real))

println("v sim min/max = ", minimum(v_sim), " / ", maximum(v_sim))
println("v real min/max = ", minimum(v_real), " / ", maximum(v_real))

println("T sim min/max = ", minimum(T_sim), " / ", maximum(T_sim))
println("T real min/max = ", minimum(T_real), " / ", maximum(T_real))

println("S sim min/max = ", minimum(S_sim), " / ", maximum(S_sim))
println("S real min/max = ", minimum(S_real), " / ", maximum(S_real))

println("\nLatest sim profile time (days) = ", sim["saved_times"][end] / 86400)
println("Compared against GLORYS day index = 5")