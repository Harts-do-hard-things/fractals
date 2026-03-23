# --------------------------------
# Deterministic Iteration
# --------------------------------

function _deterministic_expand_points(
    points::Vector{SVector{2,Float64}},
    maps::Vector{AffineMap{Float64}},
    n::Integer
)
    nmaps = length(maps)

    newsize = length(points) * nmaps^n
    @assert newsize ≥ 0 "Integer overflow"
    @assert newsize < 10^8 "Too many points allocated"
    current = points
    for _ in 1:n
        current_len = length(current)
        next = Vector{SVector{2,Float64}}(undef, current_len * nmaps)

        @threads for i in 1:nmaps
            start = (i-1) * current_len + 1
            stop = i * current_len
            @inbounds next[start:stop] = maps[i].(current)
        end

        current = next
    end

    return current
end

function deterministic_iterate(ifs::IFS,
                               n::Integer;
                               warmup::Integer=DEFAULT_WARMUP,
                               seed::Union{Nothing,Integer}=nothing)
    base = IFS(ifs.name, ifs.docs, copy(ifs.points), ifs.maps, ifs.weights, ifs.limits)
    iterate!(base; warmup=warmup, seed=seed)

    result = _deterministic_expand_points(base.points, ifs.maps, n)
    return IFS(ifs.name, ifs.docs, result, ifs.maps, ifs.weights, ifs.limits)
end
