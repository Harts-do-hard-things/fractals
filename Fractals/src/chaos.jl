# --------------------------------
# Chaos Game Iteration
# --------------------------------

function _iterate_serial!(ifs::IFS;
                          warmup=DEFAULT_WARMUP,
                          seed::Union{Nothing,Integer}=nothing)

    maps = ifs.maps
    weights = ifs.weights
    map_indices = Base.OneTo(length(maps))

    x = SVector{2,Float64}(0.0, 0.0)

    if isnothing(seed)
        for _ in 1:warmup
            x = maps[sample(map_indices, weights)](x)
        end
        idxs = sample(map_indices, weights, length(ifs.points))
    else
        rng = MersenneTwister(seed)
        for _ in 1:warmup
            x = maps[sample(rng, map_indices, weights)](x)
        end
        idxs = sample(rng, map_indices, weights, length(ifs.points))
    end

    for i in eachindex(ifs.points)
        x = maps[idxs[i]](x)
        ifs.points[i] = x
    end

    return ifs
end

function _iterate_parallel!(ifs::IFS;
                            warmup=DEFAULT_WARMUP,
                            seed::Union{Nothing,Integer}=nothing)
    maps = ifs.maps
    weights = ifs.weights
    map_indices = Base.OneTo(length(maps))
    npts = length(ifs.points)

    @threads for tid in 1:nthreads()
        # chunk for this thread
        lo = fld((tid-1)*npts, nthreads()) + 1
        hi = fld(tid*npts, nthreads())
        lo > hi && continue

        rng = isnothing(seed) ? nothing : MersenneTwister(seed + tid - 1)

        # independent chain per thread
        x = SVector{2,Float64}(0.0, 0.0)
        if isnothing(rng)
            for _ in 1:warmup
                x = maps[sample(map_indices, weights)](x)
            end
            @inbounds for i in lo:hi
                x = maps[sample(map_indices, weights)](x)
                ifs.points[i] = x
            end
        else
            for _ in 1:warmup
                x = maps[sample(rng, map_indices, weights)](x)
            end
            @inbounds for i in lo:hi
                x = maps[sample(rng, map_indices, weights)](x)
                ifs.points[i] = x
            end
        end
    end

    return ifs
end

function iterate!(ifs::IFS;
                  warmup=DEFAULT_WARMUP,
                  seed::Union{Nothing,Integer}=nothing)
    if nthreads() > 1
        return _iterate_parallel!(ifs; warmup=warmup, seed=seed)
    end
    return _iterate_serial!(ifs; warmup=warmup, seed=seed)
end

# Backward-compatible entrypoint. Iteration is now automatically threaded via iterate!.
function iterate_parallel!(ifs::IFS;
                           warmup=DEFAULT_WARMUP,
                           seed::Union{Nothing,Integer}=nothing)
    return iterate!(ifs; warmup=warmup, seed=seed)
end
