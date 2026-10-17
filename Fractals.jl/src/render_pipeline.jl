# --------------------------------
# Render pipeline
# --------------------------------
#
# One pipeline for every method (`render(::Render)`); everything method-specific is a method of
# `draw`, `colorize`, `seed_image`, or `overlay!`. Each `draw` returns `(image, limits)`, where
# `limits` is the world-space frame of that image (it differs from `ifs.limits` for polygon-
# framed methods). Must be included after render_types.jl and every file whose kernels it calls.

_with_limits(ifs::IFS, limits) = IFS(ifs.name, ifs.maps, ifs.weights, limits)

# The raster helpers in render_transformations.jl only read `.limits` from their `ifs` argument.
_frame(limits) = (; limits)

# --------------------------------
# draw: one method per render method
# --------------------------------

draw(r::Render) = draw(r.method, r)

function draw(m::Chaos, r::Render)
    points = chaos_game(r.ifs, m.npoints; warmup = m.warmup, seed = m.seed)
    return make_image(points, r.ifs.limits; resolution = r.resolution), r.ifs.limits
end

function draw(m::PointDeterministic, r::Render)
    start = chaos_game(r.ifs, m.npoints; warmup = m.warmup, seed = m.seed)
    points = deterministic_iterate(r.ifs, start, m.iterations)
    return make_image(points, r.ifs.limits; resolution = r.resolution), r.ifs.limits
end

function draw(m::Inverse, r::Render)
    img = rasterize_image_inversely(
        r.ifs,
        m.iterations,
        r.ifs.limits;
        resolution = r.resolution,
        show_divergence_scale = m.divergence_scale,
    )
    return img, r.ifs.limits
end

function draw(m::Transformations, r::Render)
    rows, cols = r.resolution
    img = _render_transformations_image(
        r.ifs;
        width = cols,
        height = rows,
        initial_polygon_spec = m.polygon,
        polygon_limits_iterations = m.limits_iterations,
        color = r.color,
    )
    limits = _limits_from_polygon_point_iterations(
        r.ifs;
        initial_polygon_spec = m.polygon,
        polygon_limits_iterations = m.limits_iterations,
    )
    return img, limits
end

function draw(m::ImageIterate, r::Render)
    img, limits = seed_image(m.source, r)
    raster_ifs = _with_limits(r.ifs, limits)
    for pass = 1:m.iterations
        img = iterate_image(raster_ifs, img; colors = r.color && pass == m.iterations)
    end
    return img, limits
end

# --------------------------------
# seed_image: where ImageIterate starts
# --------------------------------

function seed_image(s::PolygonSource, r::Render)
    limits =
        s.frame == :ifs ? r.ifs.limits :
        _limits_from_polygon_point_iterations(
            r.ifs;
            initial_polygon_spec = s.polygon,
            polygon_limits_iterations = something(s.limits_iterations, r.method.iterations),
        )
    rows, cols = r.resolution
    img = _render_initial_polygon_seed_image(
        _frame(limits);
        width = cols,
        height = rows,
        initial_polygon_spec = s.polygon,
    )
    return _to_grayscale_matrix(img), limits
end

function seed_image(s::FileSource, r::Render)
    img = _load_image(s.path)
    _validate_iterate_image_input(img)
    record = _sidecar_path(s.path)
    limits = isfile(record) ? JLD2.load(record, "limits") : r.ifs.limits
    return _to_grayscale_matrix(img), limits
end

# Any render method can seed ImageIterate: its image (and frame) is the starting image.
function seed_image(m::RenderMethod, r::Render)
    img, limits = draw(m, r)
    return _to_grayscale_matrix(img), limits
end

# --------------------------------
# Finishing: color, overlays, alpha
# --------------------------------

# Default coloring is one colored iterate_image pass. Methods that color while drawing opt out.
colorize(::RenderMethod, r::Render, img, limits) =
    iterate_image(_with_limits(r.ifs, limits), img; colors = true)
colorize(::Transformations, ::Render, img, limits) = img
colorize(::ImageIterate, ::Render, img, limits) = img

function overlay!(img, o::BasePolygon, r::Render, limits)
    rows, cols = r.resolution
    return _overlay_initial_polygon!(
        img,
        _frame(limits);
        width = cols,
        height = rows,
        initial_polygon_spec = o.polygon,
        color = r.color,
    )
end

function overlay!(img, ::Axes, r::Render, limits)
    rows, cols = r.resolution
    return _overlay_axes!(img, _frame(limits); width = cols, height = rows)
end

function finish(r::Render, img, limits)
    r.color && (img = colorize(r.method, r, img, limits))
    # Overlay helpers draw into Gray/RGB pixels; raw heatmaps are plain Float32 matrices.
    isempty(r.overlays) || eltype(img) <: Real && (img = Gray{Float32}.(img))
    for o in r.overlays
        overlay!(img, o, r, limits)
    end
    r.alpha && (img = _apply_alpha_mask(img))
    return img
end

# --------------------------------
# Naming, saving, loading (DrWatson + JLD2)
# --------------------------------

# Which method fields go into file names (DrWatson's `savename` reads these).
DrWatson.allaccess(::Chaos) = (:npoints, :seed)
DrWatson.allaccess(::PointDeterministic) = (:npoints, :iterations)
DrWatson.allaccess(::Inverse) = (:iterations,)
DrWatson.allaccess(::Transformations) = (:polygon, :limits_iterations)
DrWatson.allaccess(::ImageIterate) = (:iterations,)

method_tag(::Chaos) = "chaos"
method_tag(::PointDeterministic) = "point_deterministic"
method_tag(::Inverse) = "inverse"
method_tag(::Transformations) = "transformations"
method_tag(m::ImageIterate) = "image_iterate-" * source_tag(m.source)
source_tag(::PolygonSource) = "polygon"
source_tag(::FileSource) = "file"
source_tag(m::RenderMethod) = method_tag(m)

function _render_name(r::Render)
    stem = savename(_savename_prefix(r.ifs.name) * "_" * method_tag(r.method), r.method)
    rows, cols = r.resolution
    flags = (r.color ? "_color" : "") * (r.alpha ? "_alpha" : "")
    return "$(stem)_$(rows)x$(cols)$(flags).png"
end

_sidecar_path(image_path::AbstractString) = first(splitext(image_path)) * ".jld2"

"""
    save_render(r::Render, img, limits; outpath = nothing) -> (outpath, record)

Write `img` as a PNG and the render record (`r` and `limits`, tagged with the Git commit by
DrWatson's `tagsave`) as a `.jld2` with the same stem. `outpath = nothing` names the PNG from the
IFS name, the method's parameters, the resolution, and the color/alpha flags.
"""
function save_render(r::Render, img, limits; outpath::Union{Nothing,AbstractString} = nothing)
    png = _resolve_output_path(outpath, _render_name(r))
    mkpath(dirname(png))
    _save_image_file(png, img)
    record = _sidecar_path(png)
    data = Dict{String,Any}("render" => r, "limits" => limits)
    if _SKIP_GIT[]  # precompile workload: no git subprocess
        DrWatson.wsave(record, data)
    else
        DrWatson.tagsave(record, data; gitpath = dirname(pkgdir(@__MODULE__)), warn = false)
    end
    return png, record
end

"""
    load_render(path) -> Render

The [`Render`](@ref) saved in a render record: the `.jld2` written by [`render`](@ref), or the
PNG next to it. Run it again with `render(load_render(path))`.
"""
function load_render(path::AbstractString)
    record = endswith(lowercase(path), ".jld2") ? path : _sidecar_path(path)
    return JLD2.load(record, "render")::Render
end

"""
    render(r::Render; outpath = nothing)
    render(ifs::IFS, method::RenderMethod; outpath = nothing, kwargs...)

Draw `r` with its method, apply color, overlays and alpha, then save the PNG and its render
record (see [`save_render`](@ref)). The second form builds the [`Render`](@ref) from `kwargs`.
Returns `(render, image, limits, outpath, record)`.
"""
function render(r::Render; outpath::Union{Nothing,AbstractString} = nothing)
    img, limits = draw(r)
    img = finish(r, img, limits)
    png, record = save_render(r, img, limits; outpath)
    return (; render = r, image = img, limits, outpath = png, record)
end

render(ifs::IFS, method::RenderMethod; outpath = nothing, kwargs...) =
    render(Render(ifs, method; kwargs...); outpath)
