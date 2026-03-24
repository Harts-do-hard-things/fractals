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

@inline function _contractive_row(a::Real, b::Real, d::Real, e::Real)
    aa = Float64(a)
    bb = Float64(b)
    dd = Float64(d)
    ee = Float64(e)
    return aa^2 + dd^2 < 1 &&
           bb^2 + ee^2 < 1 &&
           aa^2 + bb^2 + dd^2 + ee^2 < 1 + (aa * ee - dd * bb)^2
end

function _noncontractive_rows(eq::AbstractMatrix{<:Real})
    failures = NamedTuple[]
    for i in 1:size(eq, 1)
        a, b, d, e = eq[i, 1], eq[i, 2], eq[i, 3], eq[i, 4]
        _contractive_row(a, b, d, e) && continue
        push!(failures, (row=i, a=Float64(a), b=Float64(b), d=Float64(d), e=Float64(e)))
    end
    return failures
end

function _format_contractivity_failure(failure)
    return "row $(failure.row) is not contractive for coefficients (a=$(failure.a), b=$(failure.b), d=$(failure.d), e=$(failure.e))"
end

function _validate_contractivity(eq::AbstractMatrix{<:Real}; context::AbstractString="IFS equation matrix")
    failures = _noncontractive_rows(eq)
    isempty(failures) || throw(ArgumentError("$context contains non-contractive affine maps: $(_format_contractivity_failure(first(failures)))."))
    return nothing
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

function _resolve_nmaps(rng::AbstractRNG, nmaps::Integer)
    nmaps > 0 || throw(ArgumentError("nmaps must be > 0, got $nmaps"))
    return Int(nmaps)
end

function _resolve_nmaps(rng::AbstractRNG, nmaps::UnitRange{Int})
    isempty(nmaps) && throw(ArgumentError("nmaps range must be non-empty"))
    minimum(nmaps) > 0 || throw(ArgumentError("nmaps range must contain only positive values, got $nmaps"))
    return rand(rng, nmaps)
end

function _sample_contractive_row(rng::AbstractRNG; linear_range::Real=0.95, translation_range::Real=1.0, max_tries::Integer=10_000)
    linear_range > 0 || throw(ArgumentError("linear_range must be > 0, got $linear_range"))
    translation_range >= 0 || throw(ArgumentError("translation_range must be >= 0, got $translation_range"))
    max_tries > 0 || throw(ArgumentError("max_tries must be > 0, got $max_tries"))

    for _ in 1:max_tries
        a = rand(rng) * 2 * linear_range - linear_range
        b = rand(rng) * 2 * linear_range - linear_range
        d = rand(rng) * 2 * linear_range - linear_range
        e = rand(rng) * 2 * linear_range - linear_range
        _contractive_row(a, b, d, e) || continue
        c = rand(rng) * 2 * translation_range - translation_range
        f = rand(rng) * 2 * translation_range - translation_range
        return (a, b, d, e, c, f)
    end

    throw(ArgumentError("Unable to sample a contractive affine map within max_tries=$max_tries"))
end

function random_eq(;
                   nmaps::Union{Integer,UnitRange{Int}}=2:4,
                   translation_range::Real=1.0,
                   with_probs::Bool=true,
                   rng::AbstractRNG=Random.default_rng(),
                   max_tries::Integer=10_000)
    n = _resolve_nmaps(rng, nmaps)
    ncols = with_probs ? 7 : 6
    eq = Matrix{Float64}(undef, n, ncols)
    for i in 1:n
        a, b, d, e, c, f = _sample_contractive_row(rng;
                                                   translation_range=translation_range,
                                                   max_tries=max_tries)
        eq[i, 1] = a
        eq[i, 2] = b
        eq[i, 3] = d
        eq[i, 4] = e
        eq[i, 5] = c
        eq[i, 6] = f
    end
    if with_probs
        probs = rand(rng, n)
        probs ./= sum(probs)
        eq[:, 7] .= probs
    end
    return eq
end

function random_ifs(;
                    npoints::Integer=DEFAULT_SAMPLES,
                    name::AbstractString="",
                    docs::AbstractString="",
                    nmaps::Union{Integer,UnitRange{Int}}=2:4,
                    translation_range::Real=1.0,
                    with_probs::Bool=true,
                    rng::AbstractRNG=Random.default_rng(),
                    max_tries::Integer=10_000)
    eq = random_eq(; nmaps=nmaps,
                   translation_range=translation_range,
                   with_probs=with_probs,
                   rng=rng,
                   max_tries=max_tries)
    return IFS(eq; npoints=npoints, name=name, docs=docs)
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
