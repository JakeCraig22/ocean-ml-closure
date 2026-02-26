using JLD2
using Statistics
using Plots

gr()

# Load snapshot
d = load("output/snap_000297.jld2")

T = d["T"]
t = d["t"]

# Average over x and y (horizontal mean profile)
Tz = dropdims(mean(T, dims=(1,2)), dims=(1,2))

# Vertical grid in meters
Nz = length(Tz)
water_depth = 100.0
z = range(0, stop=water_depth, length=Nz)

# Create plot
p = plot(
    Tz,
    z,
    xlabel="Temperature",
    ylabel="Depth (m)",
    title="Mean Temperature Profile at t = $(round(t)) s",
    legend=false,
    yflip=true
)

display(p)

println("Press Enter to close...")
readline()
