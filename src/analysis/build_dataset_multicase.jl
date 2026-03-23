using JLD2
using Statistics
using Printf
using Base.Filesystem: mkpath
using Base: dirname

# -----------------------------
# forcing cases to combine
# -----------------------------
tx_values = [2.5e-4, 5e-4, 1e-3, 2e-3]

out_path = "data/generated/dataset_multicase.jld2"
H = 100.0

# -----------------------------
# helpers
# -----------------------------
function face_to_center_profile(w_face::AbstractVector)
    return 0.5 .* (w_face[1:end-1] .+ w_face[2:end])
end

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
        i1 = (k-1) * r + 1
        i2 = k * r
        f_lo[k] = mean(@view f_hi[i1:i2])
    end
    return f_lo
end

# -----------------------------
# storage
# -----------------------------
X_list = Vector{Array{Float64,2}}()   # each Nz x 4
y_list = Vector{Vector{Float64}}()    # each Nz
t_list = Float64[]
name_list = String[]
tx_list = Float64[]

# -----------------------------
# loop over forcing cases
# -----------------------------
for tx in tx_values
    coarse_dir = "output/multicase/coarse_tx$(tx)_24hr_Nz128"
    truth_dir  = "output/multicase/truth_tx$(tx)_24hr_Nz256"

    isdir(coarse_dir) || error("Missing coarse dir: $coarse_dir")
    isdir(truth_dir)  || error("Missing truth dir: $truth_dir")

    all_files = readdir(coarse_dir)
    snap_files = sort(filter(f -> startswith(f, "snap_") && endswith(f, ".jld2"), all_files))
    isempty(snap_files) && error("No snapshots found in $coarse_dir")

    @info "Processing forcing case" tx=tx snapshots=length(snap_files)

    for (idx, snap_name) in enumerate(snap_files)
        coarse_path = joinpath(coarse_dir, snap_name)
        truth_path  = joinpath(truth_dir, snap_name)

        if !isfile(truth_path)
            @warn "Skipping snapshot with no matching truth file" tx=tx snap_name=snap_name
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
        z_norm = z ./ H

        dudz = central_diff_1d(u_prof, z)
        dTdz = central_diff_1d(T_prof, z)

        T_truth_hi   = horiz_mean_profile(T_t)
        T_truth_filt = downsample_by_block_mean(T_truth_hi, Nz)

        residual_T = T_truth_filt .- T_prof

        # 4 features only: u, dudz, dTdz, z/H
        X = hcat(u_prof, dudz, dTdz, z_norm)

        push!(X_list, X)
        push!(y_list, residual_T)
        push!(t_list, Float64(coarse["t"]))
        push!(name_list, snap_name)
        push!(tx_list, tx)

        if idx % 50 == 0
            @info "Processed snapshots" tx=tx idx=idx
        end
    end
end

N = length(X_list)
N == 0 && error("No examples were processed.")

Nz = size(X_list[1], 1)
F  = size(X_list[1], 2)

X_all = Array{Float64,3}(undef, N, Nz, F)
y_all = Array{Float64,2}(undef, N, Nz)

for i in 1:N
    X_all[i, :, :] .= X_list[i]
    y_all[i, :]    .= y_list[i]
end

@info "Multicase dataset built" examples=N Nz=Nz features=F
@printf("Residual stats:\n")
@printf("  mean(|y|) = %.6e\n", mean(abs.(y_all)))
@printf("  max(|y|)  = %.6e\n", maximum(abs.(y_all)))

mkpath(dirname(out_path))
JLD2.@save out_path X_all y_all t_list name_list tx_list tx_values H

@info "Saved multicase dataset" out_path=out_path