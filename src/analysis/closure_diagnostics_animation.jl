# src/analysis/ocean_view_animation.jl
#
# "Ocean-looking" animation:
#   - x–z slice colored by Temperature T(x,z)
#   - velocity vectors overlaid (u,w) so you see motion
#   - compares Coarse vs MLP side-by-side (same color scale)
#
# Output: output/ocean_view_coarse_vs_mlp.mp4

using JLD2
using CairoMakie
using Statistics

coarse_dir = "output/surface_stress_tx5e-4_6hr_Nz128"
mlp_dir    = "output/mlclosure_mlp_surface_stress_tx5e-4_6hr_Nz128"

coarse_files = sort(filter(f -> endswith(f, ".jld2"), readdir(coarse_dir)))
mlp_files    = sort(filter(f -> endswith(f, ".jld2"), readdir(mlp_dir)))

N = min(length(coarse_files), length(mlp_files))
N == 0 && error("No snapshots found in coarse or mlp directory.")

depth = 100.0
Nz = 128
z  = collect(range(-depth, 0.0, length=Nz))

# -------------------------
# Helpers
# -------------------------
# w is sometimes on vertical faces (Nz+1). If so, average to centers.
function w_face_to_center(w::AbstractMatrix{<:Real})
    Nx, Nw = size(w)
    if Nw == Nz + 1
        wc = similar(w, Float64, Nx, Nz)
        @inbounds for k in 1:Nz
            wc[:, k] .= 0.5 .* (w[:, k] .+ w[:, k+1])
        end
        return wc
    elseif Nw == Nz
        return Float64.(w)
    else
        error("Unexpected w vertical size: got $Nw, expected $Nz or $(Nz+1).")
    end
end

# Quick min/max scan so both panels share the same color range (so comparisons are real).
function scan_T_minmax(dir::String, files::Vector{String}; nsample::Int=12)
    n = length(files)
    idxs = unique(round.(Int, range(1, n, length=min(nsample, n))))
    tmin = +Inf
    tmax = -Inf
    for i in idxs
        d = JLD2.load(joinpath(dir, files[i]))
        T = Float64.(d["T"][:, 1, :])  # (Nx, Nz)
        tmin = min(tmin, minimum(T))
        tmax = max(tmax, maximum(T))
    end
    return tmin, tmax
end

# -------------------------
# Color range shared across coarse + mlp
# -------------------------
cmin1, cmax1 = scan_T_minmax(coarse_dir, coarse_files)
cmin2, cmax2 = scan_T_minmax(mlp_dir, mlp_files)
Tmin = min(cmin1, cmin2)
Tmax = max(cmax1, cmax2)

# -------------------------
# Load first frame to get Nx and init observables
# -------------------------
c0 = JLD2.load(joinpath(coarse_dir, coarse_files[1]))
m0 = JLD2.load(joinpath(mlp_dir, mlp_files[1]))

Tcoarse0 = Float64.(c0["T"][:, 1, :])  # (Nx, Nz)
Tmlp0    = Float64.(m0["T"][:, 1, :])  # (Nx, Nz)

ucoarse0 = Float64.(c0["u"][:, 1, :])  # (Nx, Nz) typically
wcoarse0 = w_face_to_center(Float64.(c0["w"][:, 1, :]))

umlp0 = Float64.(m0["u"][:, 1, :])
wmlp0 = w_face_to_center(Float64.(m0["w"][:, 1, :]))

Nx = size(Tcoarse0, 1)
x  = collect(range(0.0, 1.0, length=Nx))  # domain extent is 1 (from your grid)

# Observables for Makie-safe updates
Tcoarse_obs = Observable(Tcoarse0)
Tmlp_obs    = Observable(Tmlp0)

ucoarse_obs = Observable(ucoarse0)
wcoarse_obs = Observable(wcoarse0)

umlp_obs = Observable(umlp0)
wmlp_obs = Observable(wmlp0)

# -------------------------
# Figure layout
# -------------------------
isdir("output") || mkpath("output")
out_mp4 = joinpath("output", "ocean_view_coarse_vs_mlp.mp4")

fig = Figure(size=(1500, 700))

axL = Axis(fig[1, 1], title="Coarse: Temperature + velocity vectors",
           xlabel="x (domain units)", ylabel="z (m)")
axR = Axis(fig[1, 2], title="MLP-coupled: Temperature + velocity vectors",
           xlabel="x (domain units)", ylabel="z (m)")

# Heatmaps: Makie expects size(matrix) == (length(x), length(z))
hmL = heatmap!(axL, x, z, Tcoarse_obs; colorrange=(Tmin, Tmax))
hmR = heatmap!(axR, x, z, Tmlp_obs;    colorrange=(Tmin, Tmax))

# Add one shared colorbar (so “same colors = same temps”)
Colorbar(fig[1, 3], hmL, label="Temperature")

# -------------------------
# Vector overlay (quiver)
# -------------------------
# Subsample vectors so it's not a mess
skipx = max(1, Int(floor(Nx / 16)))   # ~16 arrows across
skipz = 6                             # arrows every ~6 vertical points

xs = x[1:skipx:end]
zs = z[1:skipz:end]

# Build a grid of arrow positions (same for both panels)
Xpos = repeat(xs, inner=length(zs))
Zpos = repeat(zs, outer=length(xs))

# To grab U/W on the subsampled grid each frame:
function subsample_UW(U::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, skipx::Int, skipz::Int)
    Usub = U[1:skipx:end, 1:skipz:end]
    Wsub = W[1:skipx:end, 1:skipz:end]
    # Flatten into vectors in the same order we built Xpos/Zpos:
    # Usub is (length(xs), length(zs)) → we want x-major order
    return vec(Usub), vec(Wsub)
end

Uvec0L, Wvec0L = subsample_UW(ucoarse0, wcoarse0, skipx, skipz)
Uvec0R, Wvec0R = subsample_UW(umlp0,    wmlp0,    skipx, skipz)

UvecL = Observable(Uvec0L)
WvecL = Observable(Wvec0L)
UvecR = Observable(Uvec0R)
WvecR = Observable(Wvec0R)

# Scale arrows so they’re visible (tweak if needed)
arrow_scale = 250.0  # bigger = longer arrows

quiver!(axL, Xpos, Zpos, lift(u -> arrow_scale .* u, UvecL), lift(w -> arrow_scale .* w, WvecL);
        arrowsize=8)

quiver!(axR, Xpos, Zpos, lift(u -> arrow_scale .* u, UvecR), lift(w -> arrow_scale .* w, WvecR);
        arrowsize=8)

# -------------------------
# Record
# -------------------------
record(fig, out_mp4, 1:N; framerate=10) do i
    c = JLD2.load(joinpath(coarse_dir, coarse_files[i]))
    m = JLD2.load(joinpath(mlp_dir,    mlp_files[i]))

    Tcoarse = Float64.(c["T"][:, 1, :])
    Tmlp    = Float64.(m["T"][:, 1, :])

    ucoarse = Float64.(c["u"][:, 1, :])
    wcoarse = w_face_to_center(Float64.(c["w"][:, 1, :]))

    umlp = Float64.(m["u"][:, 1, :])
    wmlp = w_face_to_center(Float64.(m["w"][:, 1, :]))

    # Update fields
    Tcoarse_obs[] = Tcoarse
    Tmlp_obs[]    = Tmlp

    ucoarse_obs[] = ucoarse
    wcoarse_obs[] = wcoarse
    umlp_obs[]    = umlp
    wmlp_obs[]    = wmlp

    # Update quiver vectors
    Uc, Wc = subsample_UW(ucoarse, wcoarse, skipx, skipz)
    Um, Wm = subsample_UW(umlp,    wmlp,    skipx, skipz)

    UvecL[] = Uc
    WvecL[] = Wc
    UvecR[] = Um
    WvecR[] = Wm
end

println("Wrote: ", out_mp4)