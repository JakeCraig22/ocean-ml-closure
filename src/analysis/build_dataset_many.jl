# src/analysis/build_dataset_many.jl
#
# Build a training dataset from MANY snapshots.
# inputs:  u(z), dudz(z), dTdz(z) from coarse
# target:  residual_T(z) = T_truth_filtered(z) - T_coarse(z)
#
# Saves: data/generated/dataset_many.jld2
#
# Run:
#   julia --project=. src/analysis/build_dataset_many.jl

using JLD2
using Statistics
using Printf
using Base.Filesystem: mkpath
using Base: dirname

coarse_dir = "output/surface_stress_tx5e-4_6hr_Nz128"
truth_dir  = "output/truth_surface_stress_tx5e-4_6hr_Nz256"
out_path   = "data/generated/dataset_many.jld2"

#numsnaps
max_examples = nothing  


# Helpers

function horiz_mean_profile(A::AbstractArray{T,3}) where {T}
    Nz = size(A, 3)
    prof = Vector{Float64}(undef, Nz)
    @inbounds for k in 1:Nz
        prof[k] = mean(@view A[:, :, k])
    end
    return prof
end

function central_diff_1d(f::AbstractVector, z::AbstractVector)
    @assert length(f) == length(z)
    Nz = length(z)
    d = Vector{Float64}(undef, Nz)
    d[1]  = (f[2] - f[1]) / (z[2] - z[1])
    d[Nz] = (f[Nz] - f[Nz-1]) / (z[Nz] - z[Nz-1])
    @inbounds for k in 2:(Nz-1)
        d[k] = (f[k+1] - f[k-1]) / (z[k+1] - z[k-1])
    end
    return d
end

function downsample_by_block_mean(f_hi::AbstractVector, Nz_lo::Int)
    Nz_hi = length(f_hi)
    @assert Nz_hi % Nz_lo == 0 "Nz_hi must be a multiple of Nz_lo"
    r = Nz_hi ÷ Nz_lo
    f_lo = Vector{Float64}(undef, Nz_lo)
    @inbounds for k in 1:Nz_lo
        i1 = (k-1)*r + 1
        i2 = k*r
        f_lo[k] = mean(@view f_hi[i1:i2])
    end
    return f_lo
end

all_files = readdir(coarse_dir)
snap_files = sort(filter(f -> startswith(f, "snap_") && endswith(f, ".jld2"), all_files))
snap_paths = [joinpath(coarse_dir, f) for f in snap_files]
if isempty(snap_paths)
    error("No snapshots found in $coarse_dir")
end

if max_examples !== nothing
    snap_paths = snap_paths[1:min(end, max_examples)]
end

@info "Found snapshots" count=length(snap_paths)


# Preallocate storage (after first example to learn Nz)
H = 100.0  

X_list = Vector{Array{Float64,2}}()  # each is Nz x 3
y_list = Vector{Vector{Float64}}()  # each is Nz
t_list = Float64[]
name_list = String[]


# Loop snapshots
for (idx, coarse_path) in enumerate(snap_paths)
    snap_name = splitpath(coarse_path)[end]
    truth_path = joinpath(truth_dir, snap_name)

    if !isfile(truth_path)
        @warn "Skipping (no matching truth snapshot)" snap_name truth_path
        continue
    end

    coarse = JLD2.load(coarse_path)
    truth  = JLD2.load(truth_path)

    u_c = coarse["u"]
    T_c = coarse["T"]
    T_t = truth["T"]

    u_prof = horiz_mean_profile(u_c)
    T_prof = horiz_mean_profile(T_c)

    Nz = length(u_prof)
    z = collect(range(-H, 0.0, length=Nz))

    dudz = central_diff_1d(u_prof, z)
    dTdz = central_diff_1d(T_prof, z)

    T_truth_hi = horiz_mean_profile(T_t)
    T_truth_filt = downsample_by_block_mean(T_truth_hi, Nz)

    residual_T = T_truth_filt .- T_prof

    X = hcat(u_prof, dudz, dTdz)  # Nz x 3

    push!(X_list, X)
    push!(y_list, residual_T)
    push!(t_list, Float64(coarse["t"]))
    push!(name_list, snap_name)

    if idx % 25 == 0
        @info "Processed" idx snap_name
    end
end

N = length(X_list)
if N == 0
    error("No matched snapshot pairs were processed. Check directories and filenames.")
end

Nz = size(X_list[1], 1)

# Stack into arrays: X (N, Nz, 3), y (N, Nz)
X_all = Array{Float64,3}(undef, N, Nz, 3)
y_all = Array{Float64,2}(undef, N, Nz)

for i in 1:N
    X_all[i, :, :] .= X_list[i]
    y_all[i, :]    .= y_list[i]
end

@info "Dataset built" examples=N Nz=Nz
@printf("Residual stats over dataset:\n")
@printf("  mean(|y|) = %.6e\n", mean(abs.(y_all)))
@printf("  max(|y|)  = %.6e\n", maximum(abs.(y_all)))

mkpath(dirname(out_path))
JLD2.@save out_path X_all y_all t_list name_list coarse_dir truth_dir H

@info "Saved dataset" out_path