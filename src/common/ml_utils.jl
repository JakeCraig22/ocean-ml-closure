module CommonMLUtils

using Random

export make_batches, make_batches_rows, poly2_features

function make_batches(X, y; batchsize=2048, rng=MersenneTwister(42))
    M = size(X, 2)
    order = randperm(rng, M)
    batches = Vector{Tuple{Matrix{Float32}, Vector{Float32}}}()
    j = 1
    while j <= M
        sel = order[j:min(j + batchsize - 1, M)]
        push!(batches, (X[:, sel], y[sel]))
        j += batchsize
    end
    return batches
end

function make_batches_rows(X, y; batchsize=2048, rng=MersenneTwister(42))
    N = size(X, 1)
    idx = Random.randperm(rng, N)
    batches = []
    for i in 1:batchsize:N
        j = min(i + batchsize - 1, N)
        rows = idx[i:j]
        push!(batches, (X[rows, :]', reshape(y[rows], 1, :)))
    end
    return batches
end

function poly2_features(X::Array{Float64,2})
    M, F = size(X)
    cols = Vector{Vector{Float64}}()

    for i in 1:F
        push!(cols, X[:, i])
    end

    for i in 1:F
        push!(cols, X[:, i].^2)
    end

    for i in 1:F
        for j in (i + 1):F
            push!(cols, X[:, i] .* X[:, j])
        end
    end

    return hcat(cols...)
end

end