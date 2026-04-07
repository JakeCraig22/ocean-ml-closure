using JLD2
using Statistics

no_qt = load("output/era5/ml_forced_30day_glorysinit_scaledx005_scaledy010.jld2")
qt    = load("output/era5/ml_forced_30day_glorysinit_scaledx005_scaledy010_qt.jld2")
glorys = load("data/generated/glorys_processed.jld2")

rmse(a, b) = sqrt(mean((a .- b).^2))

u_noqt = no_qt["u_profiles"][end]
v_noqt = no_qt["v_profiles"][end]
T_noqt = no_qt["T_profiles"][end]
S_noqt = no_qt["S_profiles"][end]

u_qt = qt["u_profiles"][end]
v_qt = qt["v_profiles"][end]
T_qt = qt["T_profiles"][end]
S_qt = qt["S_profiles"][end]

u_real = glorys["u_profiles"][30]
v_real = glorys["v_profiles"][30]
T_real = glorys["T_profiles"][30]
S_real = glorys["S_profiles"][30]

println("---- FINAL COMPARISON (DAY 30) ----")

println("\nU RMSE:")
println("No QT = ", rmse(u_noqt, u_real))
println("QT    = ", rmse(u_qt,   u_real))

println("\nV RMSE:")
println("No QT = ", rmse(v_noqt, v_real))
println("QT    = ", rmse(v_qt,   v_real))

println("\nT RMSE:")
println("No QT = ", rmse(T_noqt, T_real))
println("QT    = ", rmse(T_qt,   T_real))

println("\nS RMSE:")
println("No QT = ", rmse(S_noqt, S_real))
println("QT    = ", rmse(S_qt,   S_real))

println("\nTemperature ranges:")
println("No QT min/max = ", minimum(T_noqt), " / ", maximum(T_noqt))
println("QT    min/max = ", minimum(T_qt),   " / ", maximum(T_qt))
println("REAL  min/max = ", minimum(T_real), " / ", maximum(T_real))