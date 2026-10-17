# --------------------------------
# Chaos Game Iteration
# --------------------------------

function _chaos_serial(
    ifs::IFS,
    n::Integer;
    warmup = DEFAULT_WARMUP,
    seed::Union{Nothing,Integer} = nothing,
)
    maps = ifs.maps
    weights = ifs.weights
    map_indices = Base.OneTo(length(maps))
    points = Vector{SVector{2,Float64}}(undef, n)

    x = SVector{2,Float64}(0.0, 0.0)

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    for _ = 1:warmup
        x = maps[sample(rng, map_indices, weights)](x)
    end
    idxs = sample(rng, map_indices, weights, n)

    for i = 1:n
        x = maps[idxs[i]](x)
        points[i] = x
    end

    return points
end

function _chaos_parallel(
    ifs::IFS,
    n::Integer;
    warmup = DEFAULT_WARMUP,
    seed::Union{Nothing,Integer} = nothing,
)
    maps = ifs.maps
    weights = ifs.weights
    map_indices = Base.OneTo(length(maps))
    points = Vector{SVector{2,Float64}}(undef, n)

    @threads for tid = 1:nthreads()
        # chunk for this thread
        lo = fld((tid - 1) * n, nthreads()) + 1
        hi = fld(tid * n, nthreads())
        lo > hi && continue

        rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed + tid - 1)

        # independent chain per thread
        x = SVector{2,Float64}(0.0, 0.0)
        for _ = 1:warmup
            x = maps[sample(rng, map_indices, weights)](x)
        end
        @inbounds for i = lo:hi
            x = maps[sample(rng, map_indices, weights)](x)
            points[i] = x
        end
    end

    return points
end

"""
    chaos_game(ifs, n=DEFAULT_SAMPLES; warmup=DEFAULT_WARMUP, seed=nothing) -> Vector{SVector{2,Float64}}

Run the chaos game and return `n` points: starting from the origin, discard `warmup` steps,
then record one point per step by applying a map chosen by `ifs.weights`.

Threaded automatically when Julia runs with more than one thread (each thread fills a chunk
of points). Pass an integer `seed` for reproducible output; results are deterministic for a
fixed seed, `n`, and thread count.
"""
function chaos_game(
    ifs::IFS,
    n::Integer = DEFAULT_SAMPLES;
    warmup = DEFAULT_WARMUP,
    seed::Union{Nothing,Integer} = nothing,
)
    n >= 0 || throw(ArgumentError("n must be >= 0, got $n"))
    if nthreads() > 1
        return _chaos_parallel(ifs, n; warmup = warmup, seed = seed)
    end
    return _chaos_serial(ifs, n; warmup = warmup, seed = seed)
end
