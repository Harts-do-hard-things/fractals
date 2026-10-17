# --------------------------------
# Chaos Game Iteration
# --------------------------------

function _iterate_serial!(ifs::IFS; warmup = DEFAULT_WARMUP, seed::Union{Nothing,Integer} = nothing)

    maps = ifs.maps
    weights = ifs.weights
    map_indices = Base.OneTo(length(maps))

    x = SVector{2,Float64}(0.0, 0.0)

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    for _ = 1:warmup
        x = maps[sample(rng, map_indices, weights)](x)
    end
    idxs = sample(rng, map_indices, weights, length(ifs.points))

    for i in eachindex(ifs.points)
        x = maps[idxs[i]](x)
        ifs.points[i] = x
    end

    return ifs
end

function _iterate_parallel!(
    ifs::IFS;
    warmup = DEFAULT_WARMUP,
    seed::Union{Nothing,Integer} = nothing,
)
    maps = ifs.maps
    weights = ifs.weights
    map_indices = Base.OneTo(length(maps))
    npts = length(ifs.points)

    @threads for tid = 1:nthreads()
        # chunk for this thread
        lo = fld((tid - 1) * npts, nthreads()) + 1
        hi = fld(tid * npts, nthreads())
        lo > hi && continue

        rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed + tid - 1)

        # independent chain per thread
        x = SVector{2,Float64}(0.0, 0.0)
        for _ = 1:warmup
            x = maps[sample(rng, map_indices, weights)](x)
        end
        @inbounds for i = lo:hi
            x = maps[sample(rng, map_indices, weights)](x)
            ifs.points[i] = x
        end
    end

    return ifs
end

"""
    iterate!(ifs; warmup=DEFAULT_WARMUP, seed=nothing) -> ifs

Run the chaos game in place, filling `ifs.points`: starting from the origin, discard
`warmup` steps, then record one point per step by applying a map chosen by `ifs.weights`.

Threaded automatically when Julia runs with more than one thread (each thread fills a chunk
of points). Pass an integer `seed` for reproducible output; results are deterministic for a
fixed seed, input, and thread count. `ifs.limits` is not changed (see
[`refresh_limits`](@ref)).
"""
function iterate!(ifs::IFS; warmup = DEFAULT_WARMUP, seed::Union{Nothing,Integer} = nothing)
    if nthreads() > 1
        return _iterate_parallel!(ifs; warmup = warmup, seed = seed)
    end
    return _iterate_serial!(ifs; warmup = warmup, seed = seed)
end

"""
    iterate_parallel!(ifs; warmup=DEFAULT_WARMUP, seed=nothing) -> ifs

Compatibility alias for [`iterate!`](@ref), which is already threaded.
"""
function iterate_parallel!(
    ifs::IFS;
    warmup = DEFAULT_WARMUP,
    seed::Union{Nothing,Integer} = nothing,
)
    return iterate!(ifs; warmup = warmup, seed = seed)
end
