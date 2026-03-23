# --------------------------------
# Configuration
# --------------------------------

const RESOLUTION = (1504, 2256)
const DEFAULT_WARMUP = 50
const DEFAULT_SAMPLES = 1_000_000
const DEFAULT_MEDIA_DIR = "media"
const _DEFAULT_INITIAL_POLYGON_LIMITS = ((0.0, 1.0), (0.0, 1.0))
const _DEFAULT_INITIAL_POLYGON_SEGMENTS = [
    (SVector{2,Float64}(0.0, 1.0), SVector{2,Float64}(1.0, 1.0)),
    (SVector{2,Float64}(1.0, 1.0), SVector{2,Float64}(1.0, 0.0)),
    (SVector{2,Float64}(1.0, 0.0), SVector{2,Float64}(0.0, 0.0)),
    (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, 1.0)),
    (SVector{2,Float64}(1/6, 5/6), SVector{2,Float64}(1/6, 1/6)),
    (SVector{2,Float64}(1/6, 1/6), SVector{2,Float64}(5/9, 1/6)),
]
const _EQUILATERAL_TRIANGLE_LIMITS = ((0.0, 1.0), (0.0, sqrt(3.0) / 2.0))
const _EQUILATERAL_TRIANGLE_SEGMENTS = [
    (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)),
    (SVector{2,Float64}(1.0, 0.0), SVector{2,Float64}(0.5, sqrt(3.0) / 2.0)),
    (SVector{2,Float64}(0.5, sqrt(3.0) / 2.0), SVector{2,Float64}(0.0, 0.0)),
]
const _LINE_BASE_LIMITS = ((0.0, 1.0), (-0.1, 0.1))
const _LINE_ARROW_SEGMENTS = [
    (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)),
    (SVector{2,Float64}(0.5, 0.0), SVector{2,Float64}(0.45, 0.05)),
    (SVector{2,Float64}(0.5, 0.0), SVector{2,Float64}(0.45, -0.05)),
]
const _LINE_SEGMENTS = [
    (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)),
]
const _INITIAL_POLYGON_NAMES = (:default, :equilateral_triangle, :line_arrow, :line)

@enum RenderMethod begin
    Chaos
    Parallel
    PointDeterministic
    ImageIterate
    Inverse
    RenderTransformations
end

const _RENDER_METHOD_CHOICES = (
    Chaos,
    Parallel,
    PointDeterministic,
    ImageIterate,
    Inverse,
    RenderTransformations,
)
