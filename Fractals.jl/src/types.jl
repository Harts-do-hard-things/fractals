# --------------------------------
# Affine Map
# --------------------------------

struct AffineMap{T<:AbstractFloat}
    A::SMatrix{2,2,T,4}
    b::SVector{2,T}
end

(m::AffineMap)(x::SVector{2,T}) where T = m.A * x + m.b

function Base.show(io::IO, m::AffineMap)
    println(io, "AffineMap:")
    println(io, "  [", @sprintf("% .5f", m.A[1,1]), "  ", @sprintf("% .5f", m.A[1,2]), "] [x]   [", @sprintf("% .5f", m.b[1]), "]")
    println(io, "  [", @sprintf("% .5f", m.A[2,1]), "  ", @sprintf("% .5f", m.A[2,2]), "] [y] + [", @sprintf("% .5f", m.b[2]), "]")
end

Base.inv(m::AffineMap{T}) where T = begin
    Ainv = inv(m.A)
    AffineMap(Ainv, -Ainv * m.b)
end

AffineMap(A::AbstractMatrix{T},
          b::AbstractVector{T}) where {T<:AbstractFloat} =
    AffineMap(
        SMatrix{2,2,T}(A),
        SVector{2,T}(b)
    )
AffineMap(a11::T,a12::T,
          a21::T,a22::T,
          b1::T,b2::T) where {T<:AbstractFloat} =
    AffineMap(
        SMatrix{2,2,Float64,4}((a11,a21,a12,a22)),
        SVector{2,Float64}(b1,b2)
    )

# --------------------------------
# IFS Type
# --------------------------------

struct IFS
    name::String
    docs::String
    points::Vector{SVector{2,Float64}} # TODO make this parameterizable
    maps::Vector{AffineMap{Float64}}
    weights::Weights{Float64,Float64,Vector{Float64}}
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}}
end

function Base.show(io::IO, f::IFS)
    println(io, "IFS: $(f.name)")
    if !isempty(f.docs)
        println(io, "Docs:")
        for line in split(f.docs, '\n')
            println(io, "  ", line)
        end
    end
    println(io, "Maps ($(length(f.maps))):")
    for (i, m) in enumerate(f.maps)
        println(io, "  [$i]")
        show(io, m)
    end
    println(io, "Points: $(length(f.points))")
end

# --------------------------------
# Build maps from equation matrix
# --------------------------------

function _build_maps_and_weights(eq::AbstractMatrix{<:Real})
    n = size(eq, 1)
    maps = Vector{AffineMap{Float64}}(undef, n)
    probs = Vector{Float64}(undef, n)

    has_probs = size(eq, 2) == 7

    for i in 1:n
        a11,a12,a21,a22,b1,b2 = Float64.(eq[i,1:6])
        maps[i] = AffineMap(a11,a12,a21,a22,b1,b2)

        if has_probs
            probs[i] = Float64(eq[i,7])
        else
            A = SMatrix{2,2,Float64,4}((a11,a12,a21,a22))
            probs[i] = abs(det(A))
        end
    end

    return maps, Weights(probs)
end

function _validate_eq_matrix(eq::AbstractMatrix{<:Real})
    nrows, ncols = size(eq)
    nrows > 0 || throw(ArgumentError("IFS equation matrix must have at least one row, got size $(size(eq))"))
    (ncols == 6 || ncols == 7) || throw(ArgumentError("IFS equation matrix must have exactly 6 or 7 columns, got $ncols"))

    all(isfinite, eq) || throw(ArgumentError("IFS equation matrix contains non-finite values (NaN or Inf)"))

    if ncols == 7
        probs = eq[:, 7]
        all(p -> p >= 0, probs) || throw(ArgumentError("IFS probability column (7th column) must be nonnegative"))
        sum(probs) > 0 || throw(ArgumentError("IFS probability column (7th column) must have positive total weight"))
    end

    return nothing
end

# --------------------------------
# Compute limits via sampling
# --------------------------------

function compute_limits(
    maps,
    weights;
    warmup::Integer=DEFAULT_WARMUP,
    n::Integer=10_000,
    seed::Integer=0,
)
    warmup >= 0 || throw(ArgumentError("warmup must be >= 0, got $warmup"))
    n > 0 || throw(ArgumentError("n must be > 0, got $n"))

    map_indices = Base.OneTo(length(maps))
    rng = MersenneTwister(seed)
    x = SVector{2,Float64}(0.0, 0.0)

    for _ in 1:warmup
        x = maps[sample(rng, map_indices, weights)](x)
    end

    xmin = Inf; xmax = -Inf
    ymin = Inf; ymax = -Inf

    for _ in 1:n
        x = maps[sample(rng, map_indices, weights)](x)
        xx, yy = x
        xmin = min(xmin, xx)
        xmax = max(xmax, xx)
        ymin = min(ymin, yy)
        ymax = max(ymax, yy)
    end

    dx = xmax - xmin
    dy = ymax - ymin
    m = max(dx, dy)
    pad = 0.05m

    cx = (xmin + xmax)/2
    cy = (ymin + ymax)/2
    half = (m + 2pad)/2

    return ((cx-half, cx+half),
            (cy-half, cy+half))
end

function compute_limits(points::AbstractVector{<:SVector{2,Float64}})
    isempty(points) && return initial_polygon().limits

    xmin = Inf; xmax = -Inf
    ymin = Inf; ymax = -Inf
    @inbounds for p in points
        x, y = p
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
    end

    dx = xmax - xmin
    dy = ymax - ymin
    m = max(dx, dy)
    m = m == 0 ? 1e-9 : m
    pad = 0.05m

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    half = (m + 2pad) / 2

    return ((cx - half, cx + half),
            (cy - half, cy + half))
end

function compute_limits(
    ifs::IFS;
    source::Symbol=:maps,
    warmup::Integer=DEFAULT_WARMUP,
    n::Integer=10_000,
    seed::Integer=0,
)
    if source == :maps
        return compute_limits(ifs.maps, ifs.weights; warmup=warmup, n=n, seed=seed)
    elseif source == :points
        return compute_limits(ifs.points)
    end
    throw(ArgumentError("Invalid limits source '$source'. Supported: :maps, :points"))
end

function refresh_limits(
    ifs::IFS;
    source::Symbol=:maps,
    warmup::Integer=DEFAULT_WARMUP,
    n::Integer=10_000,
    seed::Integer=0,
)
    limits = compute_limits(ifs; source=source, warmup=warmup, n=n, seed=seed)
    return IFS(ifs.name, ifs.docs, copy(ifs.points), ifs.maps, ifs.weights, limits)
end

# --------------------------------
# Constructors
# --------------------------------

function IFS(eq::AbstractMatrix{<:Real};
             npoints=DEFAULT_SAMPLES,
             name::AbstractString="",
             docs::AbstractString="")

    _validate_eq_matrix(eq)
    maps, weights = _build_maps_and_weights(eq)
    limits = compute_limits(maps, weights)

    points = [SVector{2,Float64}(0.0,0.0)
              for _ in 1:npoints]

    return IFS(String(name), String(docs), points, maps, weights, limits)
end

function IFS(maps::Vector{AffineMap{Float64}},
             weights::Weights;
             npoints::Integer=DEFAULT_SAMPLES,
             name::AbstractString="",
             docs::AbstractString="",
             limits=compute_limits(maps, weights))
    points = [SVector{2,Float64}(0.0,0.0)
              for _ in 1:npoints]
    return IFS(String(name), String(docs), points, maps, weights, limits)
end
