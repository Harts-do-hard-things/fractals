# --------------------------------
# Configuration
# --------------------------------

"Default raster size `(rows, cols)` (height × width in pixels) for renders and images."
const RESOLUTION = (1504, 2256)

"Default number of chaos-game steps discarded before points are recorded."
const DEFAULT_WARMUP = 50

"Default point-cloud size for [`IFS`](@ref) construction and [`render`](@ref) inputs."
const DEFAULT_SAMPLES = 1_000_000

"""
    InitialPolygonPreset

A named seed shape for transformation renders and polygon-seeded `ImageIterate`: its `name`,
line `segments` (pairs of `SVector{2,Float64}` endpoints), and default world `limits`. Look
presets up with [`initial_polygon`](@ref); [`supported_initial_polygons`](@ref) lists them.
"""
struct InitialPolygonPreset
    name::Symbol
    segments::Vector{Tuple{SVector{2,Float64},SVector{2,Float64}}}
    limits::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}
end

const _INITIAL_POLYGON_REGISTRY = (
    InitialPolygonPreset(
        :default,
        [
            (SVector{2,Float64}(0.0, 1.0), SVector{2,Float64}(1.0, 1.0)),
            (SVector{2,Float64}(1.0, 1.0), SVector{2,Float64}(1.0, 0.0)),
            (SVector{2,Float64}(1.0, 0.0), SVector{2,Float64}(0.0, 0.0)),
            (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, 1.0)),
            (SVector{2,Float64}(1 / 6, 5 / 6), SVector{2,Float64}(1 / 6, 1 / 6)),
            (SVector{2,Float64}(1 / 6, 1 / 6), SVector{2,Float64}(5 / 9, 1 / 6)),
        ],
        ((0.0, 1.0), (0.0, 1.0)),
    ),
    InitialPolygonPreset(
        :equilateral_triangle,
        [
            (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)),
            (SVector{2,Float64}(1.0, 0.0), SVector{2,Float64}(0.5, sqrt(3.0) / 2.0)),
            (SVector{2,Float64}(0.5, sqrt(3.0) / 2.0), SVector{2,Float64}(0.0, 0.0)),
        ],
        ((0.0, 1.0), (0.0, sqrt(3.0) / 2.0)),
    ),
    InitialPolygonPreset(
        :line_arrow,
        [
            (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)),
            (SVector{2,Float64}(0.5, 0.0), SVector{2,Float64}(0.45, 0.05)),
            (SVector{2,Float64}(0.5, 0.0), SVector{2,Float64}(0.45, -0.05)),
        ],
        ((0.0, 1.0), (-0.1, 0.1)),
    ),
    InitialPolygonPreset(
        :line,
        [(SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0))],
        ((0.0, 1.0), (-0.1, 0.1)),
    ),
)

"""
    RenderMethod

Render methods accepted by [`render`](@ref)'s `method` keyword: [`Chaos`](@ref),
[`Parallel`](@ref), [`PointDeterministic`](@ref), [`ImageIterate`](@ref), [`Inverse`](@ref),
and [`RenderTransformations`](@ref). `render` also accepts the equivalent lowercase symbols
or strings (`:chaos`, `"point_deterministic"`, ...).
"""
@enum RenderMethod begin
    Chaos
    Parallel
    PointDeterministic
    ImageIterate
    Inverse
    RenderTransformations
end

@doc "Chaos-game point cloud rasterized to a heatmap ([`render_chaos`](@ref))." Chaos
@doc "Compatibility alias for [`Chaos`](@ref); `iterate!` is already threaded. May be deprecated." Parallel
@doc "Deterministic expansion of every map applied to every point ([`render_point_deterministic`](@ref))." PointDeterministic
@doc "IFS applied to a seed image rather than points ([`render_image_iterate`](@ref))." ImageIterate
@doc "Per-pixel inverse-map coverage ([`render_inverse`](@ref))." Inverse
@doc "The affine maps applied to an initial polygon ([`render_transformations`](@ref))." RenderTransformations

const _RENDER_METHOD_CHOICES =
    (Chaos, Parallel, PointDeterministic, ImageIterate, Inverse, RenderTransformations)
