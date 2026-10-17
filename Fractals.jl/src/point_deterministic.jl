# --------------------------------
# Deterministic Iteration
# --------------------------------

function _deterministic_expand_points(
    points::Vector{SVector{2,Float64}},
    maps::Vector{AffineMap{Float64}},
    n::Integer,
)
    nmaps = length(maps)

    newsize = length(points) * nmaps^n
    @assert newsize ≥ 0 "Integer overflow"
    @assert newsize < 10^8 "Too many points allocated"
    current = points
    for _ = 1:n
        current_len = length(current)
        next = Vector{SVector{2,Float64}}(undef, current_len * nmaps)

        @threads for i = 1:nmaps
            start = (i - 1) * current_len + 1
            stop = i * current_len
            @inbounds next[start:stop] = maps[i].(current)
        end

        current = next
    end

    return current
end

"""
    deterministic_iterate(ifs, points, n) -> Vector{SVector{2,Float64}}

Apply every map of `ifs` to every point of `points`, `n` times, and return the resulting
points (the input is not mutated). Each round maps the same snapshot of points through all
maps. Seed the expansion with, for example, `chaos_game(ifs, 100)`.

The point count grows as `length(points) * length(ifs.maps)^n`, and the expansion asserts the
result stays below `10^8` points, so keep `points` small for large `n`.
"""
function deterministic_iterate(ifs::IFS, points::AbstractVector{<:SVector{2,Float64}}, n::Integer)
    return _deterministic_expand_points(Vector{SVector{2,Float64}}(points), ifs.maps, n)
end
