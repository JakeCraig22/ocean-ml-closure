# src/analysis/ocean_view_animation.jl
#
# "Ocean-looking" animation:
#   - x–z slice colored by Temperature T(x,z)
#   - velocity vectors overlaid as line segments (u,w) so you see motion
#   - compares Coarse vs MLP side-by-side (same color scale)
#
# Output:
#   output/ocean_view_coarse_vs_mlp.mp4   (or .gif if you change EXT below)

using JLD2
using CairoMakie
using Statistics

# -------------------------
# Directories (from previous sims)
# -------------------------
coarse_dir = "output/surface_stress_tx5e-4_6hr_Nz128"
mlp_dir    = "output/mlclosure_mlp_surface_stress_tx5e-4_6hr_Nz128"

coarse_files = sort(filter(f -> endswith(f, ".jld2"), readdir(coarse_dir)))
mlp_files    = sort(filter(f -> endswith(f, ".jld2"), readdir(mlp_dir)))

N = min(length(coarse_files), length(mlp_files))
N == 0 && error("No snapshots found in coarse or mlp directory.")

# -------------------------
# Grid info (matches your sims)
# -------------------------
depth = 100.0
Nz = 128
z  = collect(range(-depth, 0.0, length=Nz))

# -------------------------
# Helpers
# -------------------------
function w_face_to_center(w::AbstractMatrix{<:Real}, Nz::Int)
    Nx, Nw = size(w)
    if Nw == Nz + 1
        wc = Array{Float64}(undef, Nx, Nz)
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

function subsample_UW(U::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, skipx::Int, skipz::Int)
    Usub = U[1:skipx:end, 1:skipz:end]
    Wsub = W[1:skipx:end, 1:skipz:end]
    return Usub, Wsub
end

# Build arrow segments as line endpoints:
# returns Vector{Point2f} pairs, length = 2 * (nx*nz)
function build_segments(xs::AbstractVector, zs::AbstractVector,
                        Usub::AbstractMatrix{<:Real}, Wsub::AbstractMatrix{<:Real},
                        arrow_scale::Real)
    nx = size(Usub, 1)
    nz = size(Usub, 2)
    @assert nx == length(xs)
    @assert nz == length(zs)

    pts = Vector{Point2f}(undef, 2 * nx * nz)
    idx = 1
    @inbounds for ix in 1:nx
        x0 = xs[ix]
        for iz in 1:nz
            z0 = zs[iz]
            x1 = x0 + arrow_scale * Usub[ix, iz]
            z1 = z0 + arrow_scale * Wsub[ix, iz]
            pts[idx] = Point2f(x0, z0); idx += 1
            pts[idx] = Point2f(x1, z1); idx += 1
        end
    end
    return pts
end

# -------------------------
# Output settings
# -------------------------
isdir("output") || mkpath("output")

EXT = ".mp4"   # ".mp4" or ".gif"
out_path = joinpath("output", "ocean_view_coarse_vs_mlp" * EXT)

# -------------------------
# Shared temperature color scale
# -------------------------
cmin1, cmax1 = scan_T_minmax(coarse_dir, coarse_files)
cmin2, cmax2 = scan_T_minmax(mlp_dir, mlp_files)
Tmin = min(cmin1, cmin2)
Tmax = max(cmax1, cmax2)

# -------------------------
# Load first frame (init)
# -------------------------
c0 = JLD2.load(joinpath(coarse_dir, coarse_files[1]))
m0 = JLD2.load(joinpath(mlp_dir,    mlp_files[1]))

Tcoarse0 = Float64.(c0["T"][:, 1, :])  # (Nx, Nz)
Tmlp0    = Float64.(m0["T"][:, 1, :])  # (Nx, Nz)

ucoarse0 = Float64.(c0["u"][:, 1, :])  # (Nx, Nz)
wcoarse0 = w_face_to_center(Float64.(c0["w"][:, 1, :]), Nz)

umlp0 = Float64.(m0["u"][:, 1, :])
wmlp0 = w_face_to_center(Float64.(m0["w"][:, 1, :]), Nz)

Nx = size(Tcoarse0, 1)
x  = collect(range(0.0, 1.0, length=Nx))

# Observables for heatmaps
Tcoarse_obs = Observable(Tcoarse0)
Tmlp_obs    = Observable(Tmlp0)

# -------------------------
# Figure layout
# -------------------------
fig = Figure(size=(1600, 700))

axL = Axis(fig[1, 1], title="Coarse: T(x,z) + velocity vectors", xlabel="x", ylabel="z (m)")
axR = Axis(fig[1, 2], title="MLP: T(x,z) + velocity vectors",    xlabel="x", ylabel="z (m)")

hmL = heatmap!(axL, x, z, Tcoarse_obs; colorrange=(Tmin, Tmax))
hmR = heatmap!(axR, x, z, Tmlp_obs;    colorrange=(Tmin, Tmax))

Colorbar(fig[1, 3], hmL, label="Temperature")

# -------------------------
# Vector overlay (as line segments)
# -------------------------
skipx = max(1, Int(floor(Nx / 18)))  # ~18 vectors across
skipz = 6

xs = x[1:skipx:end]
zs = z[1:skipz:end]

arrow_scale = 250.0  # bigger -> longer arrows (tweak if needed)

Usub0L, Wsub0L = subsample_UW(ucoarse0, wcoarse0, skipx, skipz)
Usub0R, Wsub0R = subsample_UW(umlp0,    wmlp0,    skipx, skipz)

segL_obs = Observable(build_segments(xs, zs, Usub0L, Wsub0L, arrow_scale))
segR_obs = Observable(build_segments(xs, zs, Usub0R, Wsub0R, arrow_scale))

# draw segments (pairs of points) as line segments
linesegments!(axL, segL_obs; linewidth=1.5)
linesegments!(axR, segR_obs; linewidth=1.5)

# -------------------------
# Record
# -------------------------
record(fig, out_path, 1:N; framerate=10) do i
    c = JLD2.load(joinpath(coarse_dir, coarse_files[i]))
    m = JLD2.load(joinpath(mlp_dir,    mlp_files[i]))

    Tcoarse = Float64.(c["T"][:, 1, :])
    Tmlp    = Float64.(m["T"][:, 1, :])

    ucoarse = Float64.(c["u"][:, 1, :])
    wcoarse = w_face_to_center(Float64.(c["w"][:, 1, :]), Nz)

    umlp = Float64.(m["u"][:, 1, :])
    wmlp = w_face_to_center(Float64.(m["w"][:, 1, :]), Nz)

    # Update heatmaps
    Tcoarse_obs[] = Tcoarse
    Tmlp_obs[]    = Tmlp

    # Update vector segments
    UsubL, WsubL = subsample_UW(ucoarse, wcoarse, skipx, skipz)
    UsubR, WsubR = subsample_UW(umlp,    wmlp,    skipx, skipz)

    segL_obs[] = build_segments(xs, zs, UsubL, WsubL, arrow_scale)
    segR_obs[] = build_segments(xs, zs, UsubR, WsubR, arrow_scale)
end

println("Wrote: ", out_path)