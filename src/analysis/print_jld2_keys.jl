# src/analysis/print_jld2_keys.jl
using JLD2
using Printf
using Statistics
using Flux
using NNlib

function describe(x; name="")
    T = typeof(x)
    if x isa AbstractArray
        @printf("  %-20s | %-28s | size=%-14s | eltype=%s\n",
                name, string(T), string(size(x)), string(eltype(x)))
    else
        @printf("  %-20s | %-28s\n", name, string(T))
    end
end

function inspect_file(path::String)
    println("\n==============================")
    println("FILE: ", path)
    println("==============================")

    d = JLD2.load(path)

    ks = sort!(collect(Base.keys(d)))
    println("Keys: ", ks)

    for k in ks
        v = d[k]
        describe(v; name=String(k))
    end

    # Extra: if this is the dataset, print a couple key stats
    if "X_all" in ks && "y_all" in ks
        X = d["X_all"]
        y = d["y_all"]
        println("\nDataset quick check:")
        @printf("  X_all dims: %s (expected: N x Nz x 5)\n", string(size(X)))
        @printf("  y_all dims: %s (expected: N x Nz)\n", string(size(y)))
        @printf("  mean(|y|)=%.6e  max(|y|)=%.6e\n", mean(abs.(y)), maximum(abs.(y)))
    end
end

inspect_file("data/generated/dataset_many.jld2")
inspect_file("data/generated/poly2_baseline_model.jld2")
inspect_file("data/generated/mlp_baseline_model.jld2")