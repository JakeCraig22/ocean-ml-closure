# Build one training example (1 column-profile) for ML closure:
#   inputs  = [u(z), dudz(z), dTdz(z)] from coarse run
#   target  = residual_T(z) = T_truth_filtered(z) - T_coarse(z)

using JLD2
using Statistics
using Printf
using Base.Filesystem: mkpath

coarse_snap_path = "output/surface_stress_tx5e-4_6hr_Nz128/snap_004330.jld2"
truth_snap_path  = "output/truth_surface_stress_tx5e-4_6hr_Nz256/snap_004330.jld2"
out_path         = "data/generated/dataset_001.jld2"


# Helpers
"""
horiz_mean_profile(A)
Given a 3D array A[x,y,z], return 1D profile mean(A[:,:,k]) over x,y for each z.
Assumes z is the 3rd dimension.
"""
function horiz_mean_profile(A::AbstractArray{T,3}) where {T}
    Nz = size(A, 3)
    prof = Vector{Float64}(undef, Nz)
    @inbounds for k in 1:Nz
        prof[k] = mean(@view A[:, :, k])
    end
    return prof
end

"""
 central_diff_1d(f, z)
Compute df/dz on a 1D vertical grid using:
- forward diff at k=1
- backward diff at k=end
- central diff elsewhere
"""
function central_diff_1d(f::AbstractVector, z::AbstractVector)
    @assert length(f) == length(z)
    Nz = length(z)
    d = similar(Float64.(f))

    # endpoints
    d[1]  = (f[2] - f[1]) / (z[2] - z[1])
    d[Nz] = (f[Nz] - f[Nz-1]) / (z[Nz] - z[Nz-1])

    # interior
    @inbounds for k in 2:(Nz-1)
        d[k] = (f[k+1] - f[k-1]) / (z[k+1] - z[k-1])
    end
    return d
end

"""
downsample_by_block_mean(f_hi, Nz_lo)
Downsample a high-res vertical profile f_hi to Nz_lo by block-averaging.
Assumes Nz_hi is an integer multiple of Nz_lo (true for 256 -> 128).
"""
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


# Load snapshots

@info "Loading coarse snapshot" coarse_snap_path
coarse = JLD2.load(coarse_snap_path)

@info "Loading truth snapshot" truth_snap_path
truth  = JLD2.load(truth_snap_path)

# Pull fields 
u_c = coarse["u"]
T_c = coarse["T"]

T_t = truth["T"]


# Build coarse profiles
u_prof = horiz_mean_profile(u_c)
T_prof = horiz_mean_profile(T_c)

Nz_c = length(u_prof)
@assert length(T_prof) == Nz_c

# Vertical coordinate: if you stored z explicitly in file, use it.
# Otherwise assume uniform z-grid over [-H, 0] with H = 100 m
H = 100.0
z = range(-H, 0.0, length=Nz_c) |> collect

dudz = central_diff_1d(u_prof, z)
dTdz = central_diff_1d(T_prof, z)


# Compute target residual profile

T_truth_prof_hi = horiz_mean_profile(T_t)

# Downsample truth profile to coarse Nz by block-mean
T_truth_filt = downsample_by_block_mean(T_truth_prof_hi, Nz_c)

residual_T = T_truth_filt .- T_prof

@info "Dataset summary"
@printf("  Nz (coarse)           = %d\n", Nz_c)
@printf("  max |residual_T|      = %.6e\n", maximum(abs.(residual_T)))
@printf("  mean |residual_T|     = %.6e\n", mean(abs.(residual_T)))

X = hcat(u_prof, dudz, dTdz)  # Nz x 3
y = residual_T                # Nz

mkpath(dirname(out_path))

JLD2.@save out_path X y z coarse_snap_path truth_snap_path H

@info "Saved dataset" out_path