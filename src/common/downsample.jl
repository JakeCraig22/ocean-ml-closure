module CommonDownsample

using Statistics

export downsample_by_block_mean, downsample_truth, snap_list

function downsample_by_block_mean(f_hi::AbstractVector, Nz_lo::Int)
    Nz_hi = length(f_hi)
    Nz_hi % Nz_lo == 0 || error("Nz_hi must be divisible by Nz_lo")
    r = Nz_hi ÷ Nz_lo
    f_lo = Vector{Float64}(undef, Nz_lo)
    @inbounds for k in 1:Nz_lo
        i1 = (k - 1) * r + 1
        i2 = k * r
        f_lo[k] = mean(@view f_hi[i1:i2])
    end
    return f_lo
end

downsample_truth(f_hi::AbstractVector, Nz_lo::Int) = downsample_by_block_mean(f_hi, Nz_lo)

function snap_list(dir::String)
    isdir(dir) || error("Directory not found: $dir")
    files = sort(filter(f -> startswith(f, "snap_") && endswith(f, ".jld2"), readdir(dir)))
    isempty(files) && error("No snapshot files found in $dir")
    return files
end

end