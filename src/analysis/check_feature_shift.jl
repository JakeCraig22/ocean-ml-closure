using JLD2
using Statistics
using Printf

# -----------------------------
# Paths
# -----------------------------
weak_dataset_path = "data/generated/dataset_many.jld2"
strong_coarse_dir = "output/surface_stress_tx2e-3_24hr_Nz128"

H = 100.0

# -----------------------------
# Helpers
# -----------------------------
function horiz_mean_profile(A::AbstractArray{T,3}) where {T}
    Nz = size(A, 3)
    prof = Vector{Float64}(undef, Nz)
    @inbounds for k in 1:Nz
        prof[k] = mean(@view A[:, :, k])
    end
    return prof
end

function central_diff_1d(f::AbstractVector, z::AbstractVector)
    Nz = length(z)
    d = Vector{Float64}(undef, Nz)
    d[1]  = (f[2] - f[1]) / (z[2] - z[1])
    d[Nz] = (f[Nz] - f[Nz-1]) / (z[Nz] - z[Nz-1])
    @inbounds for k in 2:(Nz-1)
        d[k] = (f[k+1] - f[k-1]) / (z[k+1] - z[k-1])
    end
    return d
end

function face_to_center_profile(w_face::AbstractVector)
    return 0.5 .* (w_face[1:end-1] .+ w_face[2:end])
end

function feature_stats(x)
    return (
        minimum(x),
        maximum(x),
        mean(x),
        std(x)
    )
end

function print_stats(name, x)
    mn, mx, av, sd = feature_stats(x)
    @printf("%-8s min=% .6e  max=% .6e  mean=% .6e  std=% .6e\n", name, mn, mx, av, sd)
end

# -----------------------------
# Weak training dataset stats
# -----------------------------
println("\n==============================")
println("WEAK TRAINING DATASET")
println("==============================")

d = JLD2.load(weak_dataset_path)
X_all = d["X_all"]   # (N, Nz, 5)

u_weak    = vec(X_all[:, :, 1])
w_weak    = vec(X_all[:, :, 2])
dudz_weak = vec(X_all[:, :, 3])
dTdz_weak = vec(X_all[:, :, 4])
zn_weak   = vec(X_all[:, :, 5])

print_stats("u",    u_weak)
print_stats("w",    w_weak)
print_stats("dudz", dudz_weak)
print_stats("dTdz", dTdz_weak)
print_stats("z/H",  zn_weak)

# -----------------------------
# Strong forcing coarse run stats
# -----------------------------
println("\n==============================")
println("STRONG-FORCING COARSE RUN")
println("==============================")

files = sort(filter(f -> endswith(f, ".jld2"), readdir(strong_coarse_dir)))
isempty(files) && error("No snapshots found in $strong_coarse_dir")

u_all    = Float64[]
w_all    = Float64[]
dudz_all = Float64[]
dTdz_all = Float64[]
zn_all   = Float64[]

for f in files
    snap = JLD2.load(joinpath(strong_coarse_dir, f))

    uA = snap["u"]
    wA = snap["w"]
    TA = snap["T"]

    u_prof = horiz_mean_profile(uA)
    T_prof = horiz_mean_profile(TA)
    w_prof = horiz_mean_profile(wA)

    Nz = length(u_prof)
    z = collect(range(-H, 0.0, length=Nz))
    z_norm = z ./ H

    if length(w_prof) == Nz + 1
        w_prof = face_to_center_profile(w_prof)
    end

    dudz = central_diff_1d(u_prof, z)
    dTdz = central_diff_1d(T_prof, z)

    append!(u_all, u_prof)
    append!(w_all, w_prof)
    append!(dudz_all, dudz)
    append!(dTdz_all, dTdz)
    append!(zn_all, z_norm)
end

print_stats("u",    u_all)
print_stats("w",    w_all)
print_stats("dudz", dudz_all)
print_stats("dTdz", dTdz_all)
print_stats("z/H",  zn_all)

println("\n==============================")
println("RANGE OVERLAP CHECK")
println("==============================")

function overlap_report(name, weak, strong)
    wmin, wmax = minimum(weak), maximum(weak)
    smin, smax = minimum(strong), maximum(strong)

    println(name)
    @printf("  weak   [% .6e, % .6e]\n", wmin, wmax)
    @printf("  strong [% .6e, % .6e]\n", smin, smax)

    below = smin < wmin
    above = smax > wmax
    println("  outside weak range? ", below || above)
end

overlap_report("u",    u_weak,    u_all)
overlap_report("w",    w_weak,    w_all)
overlap_report("dudz", dudz_weak, dudz_all)
overlap_report("dTdz", dTdz_weak, dTdz_all)