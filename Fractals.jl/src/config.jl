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
