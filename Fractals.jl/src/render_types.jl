# --------------------------------
# Render description types
# --------------------------------
#
# A render is a value: `Render(ifs, method; resolution, color, alpha, overlays)`. Each render
# method is its own type carrying only the parameters it uses, and every per-method step is a
# method of a generic function (`draw`, `colorize`, `seed_image`, `overlay!`) in
# render_pipeline.jl. Constructors validate their own fields, so invalid combinations cannot be
# built and need no runtime checks later. Adding an option means adding a field to one type.

"""
    RenderMethod

Supertype of the render methods: [`Chaos`](@ref), [`PointDeterministic`](@ref),
[`Inverse`](@ref), [`Transformations`](@ref), and [`ImageIterate`](@ref).
"""
abstract type RenderMethod end

_check(cond::Bool, msg::AbstractString) = cond || throw(ArgumentError(msg))

_check_polygon(polygon::Symbol) = _check(
    polygon in supported_initial_polygons(),
    "Unknown polygon :$polygon. Supported: $(_initial_polygon_names_text())",
)

"""
    Chaos(; npoints = DEFAULT_SAMPLES, warmup = DEFAULT_WARMUP, seed = nothing)

Chaos game: `npoints` samples after `warmup` discarded steps, rasterized to a heatmap. A
`seed` makes the result reproducible (for a fixed thread count).
"""
struct Chaos <: RenderMethod
    npoints::Int
    warmup::Int
    seed::Union{Nothing,Int}
    function Chaos(npoints::Integer, warmup::Integer, seed::Union{Nothing,Integer})
        _check(npoints > 0, "npoints must be > 0, got $npoints")
        _check(warmup >= 0, "warmup must be >= 0, got $warmup")
        return new(npoints, warmup, seed)
    end
end
Chaos(; npoints = DEFAULT_SAMPLES, warmup = DEFAULT_WARMUP, seed = nothing) =
    Chaos(npoints, warmup, seed)

"""
    PointDeterministic(; npoints = 1_000, iterations = 1, warmup = DEFAULT_WARMUP, seed = nothing)

Apply every map to every point `iterations` deep, starting from `npoints` chaos-game points.
The point count grows as `npoints * nmaps^iterations`, so keep both small.
"""
struct PointDeterministic <: RenderMethod
    npoints::Int
    iterations::Int
    warmup::Int
    seed::Union{Nothing,Int}
    function PointDeterministic(
        npoints::Integer,
        iterations::Integer,
        warmup::Integer,
        seed::Union{Nothing,Integer},
    )
        _check(npoints > 0, "npoints must be > 0, got $npoints")
        _check(iterations >= 0, "iterations must be >= 0, got $iterations")
        _check(warmup >= 0, "warmup must be >= 0, got $warmup")
        return new(npoints, iterations, warmup, seed)
    end
end
PointDeterministic(; npoints = 1_000, iterations = 1, warmup = DEFAULT_WARMUP, seed = nothing) =
    PointDeterministic(npoints, iterations, warmup, seed)

"""
    Inverse(; iterations = 8, divergence_scale = true, preview = false)

Per-pixel inverse-map coverage, `iterations` deep. `divergence_scale = false` gives a binary
mask instead of graded divergence values. `preview = true` caps the tracked branch frontier
for a faster, approximate image (exact tracking otherwise).
"""
struct Inverse <: RenderMethod
    iterations::Int
    divergence_scale::Bool
    preview::Bool
    function Inverse(iterations::Integer, divergence_scale::Bool, preview::Bool)
        _check(iterations >= 0, "iterations must be >= 0, got $iterations")
        return new(iterations, divergence_scale, preview)
    end
end
Inverse(; iterations = 8, divergence_scale = true, preview = false) =
    Inverse(iterations, divergence_scale, preview)

"""
    Transformations(; polygon = :default, limits_iterations = 1)

Draw each affine map applied to the `polygon` preset (see [`initial_polygon`](@ref)), framed by
the polygon after `limits_iterations` iterations. Coloring by map follows the render's `color`.
"""
struct Transformations <: RenderMethod
    polygon::Symbol
    limits_iterations::Int
    function Transformations(polygon::Symbol, limits_iterations::Integer)
        _check_polygon(polygon)
        _check(limits_iterations >= 0, "limits_iterations must be >= 0, got $limits_iterations")
        return new(polygon, limits_iterations)
    end
end
Transformations(; polygon = :default, limits_iterations = 1) =
    Transformations(polygon, limits_iterations)

# --------------------------------
# ImageIterate and its seed sources
# --------------------------------

"""
    ImageSource

Where [`ImageIterate`](@ref) gets its starting image: a [`PolygonSource`](@ref) or a
[`FileSource`](@ref). Any [`RenderMethod`](@ref) can also be a source; its image is the seed.
"""
abstract type ImageSource end

"""
    PolygonSource(; polygon = :default, frame = :geometry, limits_iterations = nothing)

Seed with the `polygon` preset drawn in pixel space. `frame = :geometry` frames by the polygon
iterated `limits_iterations` times (default: the `ImageIterate` iteration count); `frame = :ifs`
uses the IFS's own limits.
"""
struct PolygonSource <: ImageSource
    polygon::Symbol
    frame::Symbol
    limits_iterations::Union{Nothing,Int}
    function PolygonSource(
        polygon::Symbol,
        frame::Symbol,
        limits_iterations::Union{Nothing,Integer},
    )
        _check_polygon(polygon)
        _check(frame in (:geometry, :ifs), "frame must be :geometry or :ifs, got :$frame")
        isnothing(limits_iterations) ||
            _check(limits_iterations >= 0, "limits_iterations must be >= 0")
        return new(polygon, frame, limits_iterations)
    end
end
PolygonSource(; polygon = :default, frame = :geometry, limits_iterations = nothing) =
    PolygonSource(polygon, frame, limits_iterations)

"""
    FileSource(path)

Seed with the image at `path`. When it has a render record (see [`render_record_path`](@ref)),
its saved limits frame the image; otherwise the IFS's limits are used.
"""
struct FileSource <: ImageSource
    path::String
    function FileSource(path::AbstractString)
        _check(isfile(path), "image file '$path' does not exist")
        return new(String(path))
    end
end

"""
    ImageIterate(source = PolygonSource(); iterations = 1)

Apply the IFS to an image rather than to points, `iterations` times, starting from `source`: an
[`ImageSource`](@ref) or another [`RenderMethod`](@ref), e.g.
`ImageIterate(Chaos(npoints = 10_000); iterations = 3)`.
"""
struct ImageIterate{S<:Union{ImageSource,RenderMethod}} <: RenderMethod
    source::S
    iterations::Int
    function ImageIterate(source::S, iterations::Integer) where {S<:Union{ImageSource,RenderMethod}}
        _check(iterations > 0, "iterations must be > 0, got $iterations")
        return new{S}(source, iterations)
    end
end
ImageIterate(source = PolygonSource(); iterations = 1) = ImageIterate(source, iterations)

# --------------------------------
# Overlays
# --------------------------------

"""
    Overlay

Something drawn on top of a finished image: [`BasePolygon`](@ref) or [`Axes`](@ref).
"""
abstract type Overlay end

"""
    BasePolygon(polygon = :default)

Overlay the untransformed `polygon` preset.
"""
struct BasePolygon <: Overlay
    polygon::Symbol
    function BasePolygon(polygon::Symbol = :default)
        _check_polygon(polygon)
        return new(polygon)
    end
end

"""
    Axes()

Overlay the coordinate axes.
"""
struct Axes <: Overlay end

# --------------------------------
# Output formats
# --------------------------------

"""
    OutputFormat

Supertype of the render output formats: [`PNG`](@ref) (every method) and [`SVG`](@ref) (vector
output for [`Transformations`](@ref) and `ImageIterate(PolygonSource(...))`).
"""
abstract type OutputFormat end

"""
    PNG()

Raster output: the default format of a [`Render`](@ref), supported by every render method.
"""
struct PNG <: OutputFormat end

"""
    SVG(; stroke_width = 2.0)

Vector output: the render is written as an SVG of line segments, one stroke color per map when
`color = true` (white otherwise), on a black background unless `alpha = true`. Only
[`Transformations`](@ref) and [`ImageIterate`](@ref) seeded by a [`PolygonSource`](@ref) support
it; any other method throws an `ArgumentError` when the `Render` is built.
"""
struct SVG <: OutputFormat
    stroke_width::Float64
    function SVG(stroke_width::Real)
        _check(stroke_width > 0, "stroke_width must be > 0, got $stroke_width")
        return new(stroke_width)
    end
end
SVG(; stroke_width = 2.0) = SVG(stroke_width)

# Whether a render method can be drawn in a format.
supports(::RenderMethod, ::PNG) = true
supports(::RenderMethod, ::SVG) = false
supports(::Transformations, ::SVG) = true
supports(::ImageIterate{PolygonSource}, ::SVG) = true

# --------------------------------
# The render description
# --------------------------------

"""
    Render(ifs, method; resolution = RESOLUTION, color = false, alpha = false, overlays = Overlay[],
           format = PNG())

Everything needed to produce one image: the [`IFS`](@ref), a [`RenderMethod`](@ref), the
`(rows, cols)` resolution, whether to color by map, whether zero pixels become transparent,
any [`Overlay`](@ref)s, and the [`OutputFormat`](@ref). Run it with [`render`](@ref); a saved render record loads back as
a `Render` with [`load_render`](@ref).
"""
struct Render{M<:RenderMethod,F<:OutputFormat}
    ifs::IFS
    method::M
    resolution::Tuple{Int,Int}
    color::Bool
    alpha::Bool
    overlays::Vector{Overlay}
    format::F
    function Render(
        ifs::IFS,
        method::M,
        resolution::Tuple{Integer,Integer},
        color::Bool,
        alpha::Bool,
        overlays::AbstractVector,
        format::F,
    ) where {M<:RenderMethod,F<:OutputFormat}
        _check(
            resolution[1] > 0 && resolution[2] > 0,
            "resolution must be positive, got $resolution",
        )
        _check(
            supports(method, format),
            "$(typeof(method)) cannot be rendered as $(nameof(F)); SVG supports Transformations " *
            "and ImageIterate(PolygonSource())",
        )
        rows, cols = Int(resolution[1]), Int(resolution[2])
        return new{M,F}(ifs, method, (rows, cols), color, alpha, collect(Overlay, overlays), format)
    end
end
"""
    Render(r::Render; resolution = r.resolution, color = r.color, alpha = r.alpha,
           overlays = r.overlays, ifs = r.ifs, method = r.method,
           format = r.format)

A copy of `r` with the given fields changed, e.g. `render(Render(r; resolution = (4096, 4096)))`
to re-render a saved render at a higher resolution. Point-based methods reuse their cached
points (see [`cached_points`](@ref)), so only the rasterization is redone.
"""
Render(
    r::Render;
    ifs = r.ifs,
    method = r.method,
    resolution = r.resolution,
    color = r.color,
    alpha = r.alpha,
    overlays = r.overlays,
    format = r.format,
) = Render(ifs, method, resolution, color, alpha, overlays, format)

Render(
    ifs::IFS,
    method::RenderMethod;
    resolution = RESOLUTION,
    color = false,
    alpha = false,
    overlays = Overlay[],
    format = PNG(),
) = Render(ifs, method, resolution, color, alpha, overlays, format)
