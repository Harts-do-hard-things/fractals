# --------------------------------
# Output helper
# --------------------------------

# Resolve `outpath` (auto-naming from `name_keys` of `params` when `nothing`) and safesave the
# image with source limits and full-parameter provenance. See output.jl.
function _save_render_output(
    outpath,
    img,
    ifs::IFS,
    limits;
    method::Symbol,
    params::NamedTuple,
    name_keys::Tuple,
)
    all_params = merge((; method, ifs_name = ifs.name), params)
    name_params = merge((; method), NamedTuple{name_keys}(params))
    final_outpath = _resolve_output_path(outpath, _output_name(ifs.name, name_params, "png"))
    return _save_image(final_outpath, img, limits; params = all_params)
end

# Shared argument validation for the `render_*` entry points. Checks only the keywords given.
function _check_render_args(;
    npoints::Union{Nothing,Integer} = nothing,
    warmup::Union{Nothing,Integer} = nothing,
    iterations::Union{Nothing,Integer} = nothing,
    resolution::Tuple{Int,Int},
    backend::Union{Nothing,Symbol} = nothing,
)
    isnothing(npoints) || npoints > 0 || throw(ArgumentError("npoints must be > 0, got $npoints"))
    isnothing(warmup) || warmup >= 0 || throw(ArgumentError("warmup must be >= 0, got $warmup"))
    isnothing(iterations) ||
        iterations >= 0 ||
        throw(ArgumentError("iterations must be >= 0, got $iterations"))
    resolution[1] > 0 && resolution[2] > 0 ||
        throw(ArgumentError("resolution must be positive, got $resolution"))
    isnothing(backend) ||
        backend in (:cpu, :gpu, :auto) ||
        throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))
    return nothing
end

# --------------------------------
# Method-specific render functions
# --------------------------------

"""
    render_chaos(input; npoints=nothing, warmup=DEFAULT_WARMUP, color=false, alpha=false,
                 resolution=RESOLUTION, outpath=nothing, backend=:cpu, ifs_index=nothing,
                 ifs_name=nothing, input_fn=readline)

Run the chaos game ([`iterate!`](@ref)) and rasterize the point cloud with
[`make_image`](@ref). `color=true` colors the result with one [`iterate_image`](@ref)
pass. Inputs, shared keywords, and the returned `(ifs, image, outpath, method)` tuple are
as in [`render`](@ref).
"""
function render_chaos(
    input;
    npoints::Union{Nothing,Integer} = nothing,
    warmup::Integer = DEFAULT_WARMUP,
    color::Bool = false,
    alpha::Bool = false,
    resolution::Tuple{Int,Int} = RESOLUTION,
    outpath::Union{Nothing,AbstractString} = nothing,
    backend::Symbol = :cpu,
    ifs_index::Union{Nothing,Integer} = nothing,
    ifs_name::Union{Nothing,AbstractString} = nothing,
    input_fn = readline,
)
    _check_render_args(; npoints, warmup, resolution, backend)

    ifs = _resolve_render_input(
        input;
        npoints = npoints,
        ifs_index = ifs_index,
        ifs_name = ifs_name,
        input_fn = input_fn,
    )
    iterate!(ifs; warmup = warmup)
    img = make_image(ifs; resolution = resolution, backend = backend)

    if color
        img = iterate_image(ifs, img; colors = true, backend = backend)
    end
    if alpha
        img = _apply_alpha_mask(img)
    end

    final_outpath = _save_render_output(
        outpath,
        img,
        ifs,
        ifs.limits;
        method = :chaos,
        params = (; npoints = length(ifs.points), warmup, resolution, color, alpha, backend),
        name_keys = (:npoints, :resolution, :color, :alpha),
    )
    return (ifs = ifs, image = img, outpath = final_outpath, method = :chaos)
end

"""
    render_point_deterministic(input; iterations=1, npoints=nothing, warmup=DEFAULT_WARMUP, kwargs...)

Expand the point cloud with [`deterministic_iterate`](@ref) to depth `iterations`
(`>= 0`), then rasterize it. Keep `npoints` small: the point count grows as
`npoints * nmaps^iterations`. Other keywords and the return value are as in
[`render_chaos`](@ref).
"""
function render_point_deterministic(
    input;
    npoints::Union{Nothing,Integer} = nothing,
    warmup::Integer = DEFAULT_WARMUP,
    iterations::Integer = 1,
    color::Bool = false,
    alpha::Bool = false,
    resolution::Tuple{Int,Int} = RESOLUTION,
    outpath::Union{Nothing,AbstractString} = nothing,
    backend::Symbol = :cpu,
    ifs_index::Union{Nothing,Integer} = nothing,
    ifs_name::Union{Nothing,AbstractString} = nothing,
    input_fn = readline,
)
    _check_render_args(; npoints, warmup, iterations, resolution, backend)

    ifs = _resolve_render_input(
        input;
        npoints = npoints,
        ifs_index = ifs_index,
        ifs_name = ifs_name,
        input_fn = input_fn,
    )
    rendered_ifs = deterministic_iterate(ifs, iterations; warmup = warmup)
    img = make_image(rendered_ifs; resolution = resolution, backend = backend)

    if color
        img = iterate_image(ifs, img; colors = true, backend = backend)
    end
    if alpha
        img = _apply_alpha_mask(img)
    end

    final_outpath = _save_render_output(
        outpath,
        img,
        ifs,
        rendered_ifs.limits;
        method = :point_deterministic,
        params = (;
            npoints = length(ifs.points),
            warmup,
            iterations,
            resolution,
            color,
            alpha,
            backend,
        ),
        name_keys = (:npoints, :iterations, :resolution, :color, :alpha),
    )
    return (ifs = rendered_ifs, image = img, outpath = final_outpath, method = :point_deterministic)
end

"""
    render_inverse(input; iterations=8, show_divergence_scale=true, kwargs...)

Render by inverse iteration over `ifs.limits` with [`rasterize_image_inversely`](@ref) to
depth `iterations` (`>= 0`). `show_divergence_scale=false` produces a binary mask. Other
keywords and the return value are as in [`render_chaos`](@ref).
"""
function render_inverse(
    input;
    npoints::Union{Nothing,Integer} = nothing,
    warmup::Integer = DEFAULT_WARMUP,
    iterations::Integer = 8,
    show_divergence_scale::Bool = true,
    color::Bool = false,
    alpha::Bool = false,
    resolution::Tuple{Int,Int} = RESOLUTION,
    outpath::Union{Nothing,AbstractString} = nothing,
    backend::Symbol = :cpu,
    ifs_index::Union{Nothing,Integer} = nothing,
    ifs_name::Union{Nothing,AbstractString} = nothing,
    input_fn = readline,
)
    _check_render_args(; npoints, warmup, iterations, resolution, backend)

    ifs = _resolve_render_input(
        input;
        npoints = npoints,
        ifs_index = ifs_index,
        ifs_name = ifs_name,
        input_fn = input_fn,
    )
    img = rasterize_image_inversely(
        ifs,
        iterations,
        ifs.limits;
        resolution = resolution,
        show_divergence_scale = show_divergence_scale,
        backend = backend,
    )

    if color
        img = iterate_image(ifs, img; colors = true, backend = backend)
    end
    if alpha
        img = _apply_alpha_mask(img)
    end

    final_outpath = _save_render_output(
        outpath,
        img,
        ifs,
        ifs.limits;
        method = :inverse,
        params = (; iterations, show_divergence_scale, resolution, color, alpha, backend),
        name_keys = (:iterations, :resolution, :color, :alpha),
    )
    return (ifs = ifs, image = img, outpath = final_outpath, method = :inverse)
end

"""
    render_image_iterate(input; image_source=:polygon, image_iterations=1, image_path=nothing,
                         polygon_limits_mode=:geometry, polygon_limits_iterations=nothing,
                         initial_polygon=:default, show_base=false, axis=false, kwargs...)

Build a seed image, then apply the IFS to it `image_iterations` times with
[`iterate_image`](@ref). `color=true` colors the final pass.

`image_source` picks the seed:

- `:polygon`: the [`initial_polygon`](@ref) preset rendered in pixel space. Only this source
  accepts `polygon_limits_mode` (`:geometry` frames by the polygon's iterated geometry, or
  `:ifs` by `ifs.limits`; `:default` is treated as `:geometry` with a warning),
  `polygon_limits_iterations` (defaults to `image_iterations`), and the `show_base`/`axis`
  overlays.
- `:chaos`, `:point_deterministic` (depth `iterations`), or `:inverse` (depth `iterations`,
  with `show_divergence_scale`): renders from the other methods.
- `:file`: the image at `image_path`. Its embedded source limits are used when present.

Other keywords and the return value are as in [`render_chaos`](@ref).
"""
function render_image_iterate(
    input;
    npoints::Union{Nothing,Integer} = nothing,
    warmup::Integer = DEFAULT_WARMUP,
    iterations::Integer = 1,
    color::Bool = false,
    alpha::Bool = false,
    image_source::Symbol = :polygon,
    image_path::Union{Nothing,AbstractString} = nothing,
    image_iterations::Integer = 1,
    polygon_limits_mode::Symbol = :geometry,
    polygon_limits_iterations::Union{Nothing,Integer} = nothing,
    initial_polygon::Symbol = :default,
    show_divergence_scale::Bool = true,
    show_base::Bool = false,
    axis::Bool = false,
    resolution::Tuple{Int,Int} = RESOLUTION,
    outpath::Union{Nothing,AbstractString} = nothing,
    backend::Symbol = :cpu,
    ifs_index::Union{Nothing,Integer} = nothing,
    ifs_name::Union{Nothing,AbstractString} = nothing,
    input_fn = readline,
)
    _check_render_args(; npoints, warmup, iterations, resolution, backend)
    image_iterations > 0 ||
        throw(ArgumentError("image_iterations must be > 0, got $image_iterations"))
    _validate_image_source(image_source)
    polygon_limits_mode in (:geometry, :ifs, :default) || throw(
        ArgumentError(
            "Invalid polygon_limits_mode '$polygon_limits_mode'. Supported: :geometry, :ifs, :default",
        ),
    )
    isnothing(polygon_limits_iterations) ||
        polygon_limits_iterations >= 0 ||
        throw(
            ArgumentError("polygon_limits_iterations must be >= 0, got $polygon_limits_iterations"),
        )
    preset = _resolve_initial_polygon(initial_polygon)
    image_source == :file &&
        isnothing(image_path) &&
        throw(ArgumentError("image_path is required when image_source=:file"))
    image_source != :polygon &&
        !isnothing(polygon_limits_iterations) &&
        throw(
            ArgumentError(
                "polygon_limits_iterations is only supported when image_source=:polygon for method=ImageIterate",
            ),
        )
    image_source != :polygon &&
        show_base &&
        throw(
            ArgumentError(
                "show_base is only supported when image_source=:polygon for method=ImageIterate",
            ),
        )
    image_source != :polygon &&
        axis &&
        throw(
            ArgumentError(
                "axis is only supported when image_source=:polygon for method=ImageIterate",
            ),
        )

    ifs = _resolve_render_input(
        input;
        npoints = npoints,
        ifs_index = ifs_index,
        ifs_name = ifs_name,
        input_fn = input_fn,
    )
    normalized_polygon_limits_mode =
        image_source == :polygon ? _normalize_polygon_limits_mode(polygon_limits_mode) :
        polygon_limits_mode
    resolved_polygon_limits_iterations =
        isnothing(polygon_limits_iterations) ? image_iterations : Int(polygon_limits_iterations)
    source = _resolve_image_source_context(
        ifs,
        image_source,
        image_path,
        resolution,
        warmup,
        iterations,
        iterations,
        normalized_polygon_limits_mode,
        resolved_polygon_limits_iterations,
        show_divergence_scale,
        preset,
        backend,
    )
    raster_ifs = source.ifs
    img = source.image
    for pass = 1:image_iterations
        img = iterate_image(
            raster_ifs,
            img;
            colors = (color && pass == image_iterations),
            backend = backend,
        )
    end
    if image_source == :polygon && show_base
        _overlay_initial_polygon!(
            img,
            raster_ifs;
            width = resolution[2],
            height = resolution[1],
            initial_polygon_spec = preset,
            color = color,
        )
    end
    if image_source == :polygon && axis
        _overlay_axes!(img, raster_ifs; width = resolution[2], height = resolution[1])
    end
    if alpha
        img = _apply_alpha_mask(img)
    end

    final_outpath = _save_render_output(
        outpath,
        img,
        ifs,
        raster_ifs.limits;
        method = :image_iterate,
        params = (;
            npoints = length(ifs.points),
            warmup,
            iterations,
            image_source,
            image_path,
            image_iterations,
            polygon_limits_mode = normalized_polygon_limits_mode,
            polygon_limits_iterations = resolved_polygon_limits_iterations,
            initial_polygon = preset.name,
            show_divergence_scale,
            show_base,
            axis,
            resolution,
            color,
            alpha,
            backend,
        ),
        name_keys = (:image_source, :image_iterations, :resolution, :color, :alpha),
    )
    return (ifs = raster_ifs, image = img, outpath = final_outpath, method = :image_iterate)
end

"""
    render_transformations(input; initial_polygon=:default, polygon_limits_iterations=1, show_base=false,
                           axis=false, color=true, alpha=false, resolution=RESOLUTION, outpath=nothing, ...)

Draw each affine map applied to the [`initial_polygon`](@ref) seed as a PNG, without
iterating the IFS. The frame comes from the polygon after `polygon_limits_iterations`
iterations. `show_base` overlays the untransformed polygon, and `axis` draws the coordinate
axes. Angles follow the math convention: positive is counterclockwise and `+y` is up.
Inputs and the return value are as in [`render`](@ref). For SVG output, see
[`render_transformations_svg`](@ref).
"""
function render_transformations(
    input;
    npoints::Union{Nothing,Integer} = nothing,
    show_base::Bool = false,
    axis::Bool = false,
    color::Bool = true,
    alpha::Bool = false,
    initial_polygon::Symbol = :default,
    polygon_limits_iterations::Integer = 1,
    resolution::Tuple{Int,Int} = RESOLUTION,
    outpath::Union{Nothing,AbstractString} = nothing,
    ifs_index::Union{Nothing,Integer} = nothing,
    ifs_name::Union{Nothing,AbstractString} = nothing,
    input_fn = readline,
)
    _check_render_args(; npoints, resolution)
    polygon_limits_iterations >= 0 || throw(
        ArgumentError("polygon_limits_iterations must be >= 0, got $polygon_limits_iterations"),
    )
    preset = _resolve_initial_polygon(initial_polygon)

    ifs = _resolve_render_input(
        input;
        npoints = npoints,
        ifs_index = ifs_index,
        ifs_name = ifs_name,
        input_fn = input_fn,
    )
    img = _render_transformations_image(
        ifs;
        width = resolution[2],
        height = resolution[1],
        show_base = show_base,
        initial_polygon_spec = preset,
        polygon_limits_iterations = polygon_limits_iterations,
        color = color,
        axis = axis,
        alpha = alpha,
    )
    raster_ifs = _render_transformations_effective_ifs(
        ifs;
        initial_polygon_spec = preset,
        polygon_limits_iterations = polygon_limits_iterations,
    )
    final_outpath = _save_render_output(
        outpath,
        img,
        ifs,
        raster_ifs.limits;
        method = :render_transformations,
        params = (;
            initial_polygon = preset.name,
            polygon_limits_iterations,
            show_base,
            axis,
            resolution,
            color,
            alpha,
        ),
        name_keys = (:initial_polygon, :resolution, :color, :alpha),
    )
    return (ifs = ifs, image = img, outpath = final_outpath, method = :render_transformations)
end

# --------------------------------
# Input resolution helpers
# --------------------------------

function _resolve_render_input(
    input::IFS;
    npoints::Union{Nothing,Integer},
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString},
    input_fn = readline,
)
    if !isnothing(ifs_index) || !isnothing(ifs_name)
        throw(ArgumentError("ifs_index/ifs_name are only valid for parsed string/file inputs"))
    end

    if isnothing(npoints) || length(input.points) == npoints
        return input
    end

    return IFS(
        input.maps,
        input.weights;
        npoints = npoints,
        name = input.name,
        docs = input.docs,
        limits = input.limits,
    )
end

function _resolve_render_input(
    input::AbstractMatrix{<:Real};
    npoints::Union{Nothing,Integer},
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString},
    input_fn = readline,
)
    if !isnothing(ifs_index) || !isnothing(ifs_name)
        throw(ArgumentError("ifs_index/ifs_name are only valid for parsed string/file inputs"))
    end
    resolved_npoints = isnothing(npoints) ? DEFAULT_SAMPLES : npoints
    return IFS(input; npoints = resolved_npoints)
end

function _resolve_render_input(
    input::AbstractString;
    npoints::Union{Nothing,Integer},
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString},
    input_fn = readline,  # _select_ifs_definition is defined in interactive.jl
)
    resolved_npoints = isnothing(npoints) ? DEFAULT_SAMPLES : npoints
    defs =
        isfile(input) ?
        _validated_ifs_definitions(parse_ifs_definitions_file(input); source = input) :
        _validated_ifs_definitions(parse_ifs_definitions_string(input); source = "input")

    isempty(defs) && throw(ArgumentError("No IFS definitions found in input"))
    selected = _select_ifs_definition(
        defs;
        ifs_index = ifs_index,
        ifs_name = ifs_name,
        input_fn = input_fn,
    )
    return IFS(selected.eq; npoints = resolved_npoints, name = selected.name, docs = selected.docs)
end

# --------------------------------
# High-level render entrypoint (thin delegator)
# --------------------------------

"""
    render(input; method=Chaos, npoints=nothing, warmup=DEFAULT_WARMUP, color=nothing, alpha=false,
           resolution=RESOLUTION, outpath=nothing, backend=:cpu, ifs_index=nothing, ifs_name=nothing,
           input_fn=readline, kwargs...)

Render `input` with one [`RenderMethod`](@ref) and save the image. Returns a named tuple
`(ifs, image, outpath, method)`.

`input` is an equation matrix, an [`IFS`](@ref), or `.ifs` text or a path to a `.ifs`
file. For multi-definition `.ifs` input, choose a definition with `ifs_index` or `ifs_name`.
With neither, you're asked (via `input_fn`) to confirm rendering the first definition.
Unknown selectors throw `ArgumentError` listing the available definitions.

`method` accepts the enum (`Chaos`, `PointDeterministic`, ...), a symbol, or a lowercase
string (`:point_deterministic`, `"inverse"`, ...). `Parallel` renders as `Chaos`.
Method-specific keywords go in `kwargs` and are forwarded to the matching function:
[`render_chaos`](@ref), [`render_point_deterministic`](@ref), [`render_inverse`](@ref),
[`render_image_iterate`](@ref), or [`render_transformations`](@ref).

Shared keywords:

- `npoints`: point count; `nothing` keeps an `IFS`'s count or uses `DEFAULT_SAMPLES`.
- `color`: tint by map; defaults to `false`, except `true` for `RenderTransformations`.
- `alpha`: save with transparency where the image is zero.
- `resolution`: `(rows, cols)`.
- `backend`: `:cpu`, `:gpu`, or `:auto` (see [`make_image`](@ref)).
- `outpath`: `nothing` names the file with DrWatson's `savename` under [`output_dir`](@ref);
  relative paths resolve there and absolute paths are used as given. Existing files are
  kept as `_#N` backups, and PNGs carry provenance (see [`read_provenance`](@ref)).

```julia
out = render(EISENSTEIN; method = Chaos, npoints = 250_000, resolution = (1024, 1024))
out.outpath   # .../plots/ifs_method=chaos_npoints=250000_resolution=1024x1024.png
```
"""
function render(
    input;
    method::Union{RenderMethod,Symbol,AbstractString} = Chaos,
    npoints::Union{Nothing,Integer} = nothing,
    warmup::Integer = DEFAULT_WARMUP,
    color::Union{Nothing,Bool} = nothing,
    alpha::Bool = false,
    resolution::Tuple{Int,Int} = RESOLUTION,
    outpath::Union{Nothing,AbstractString} = nothing,
    backend::Symbol = :cpu,
    ifs_index::Union{Nothing,Integer} = nothing,
    ifs_name::Union{Nothing,AbstractString} = nothing,
    input_fn = readline,
    kwargs...,
)
    parsed_method = _parse_render_method(method)
    render_method = parsed_method == Parallel ? Chaos : parsed_method

    shared_color = isnothing(color) ? false : color
    shared = (;
        npoints,
        warmup,
        color = shared_color,
        alpha,
        resolution,
        outpath,
        backend,
        ifs_index,
        ifs_name,
        input_fn,
    )

    if render_method == Chaos
        return render_chaos(input; shared..., kwargs...)
    elseif render_method == PointDeterministic
        return render_point_deterministic(input; shared..., kwargs...)
    elseif render_method == Inverse
        return render_inverse(input; shared..., kwargs...)
    elseif render_method == ImageIterate
        return render_image_iterate(input; shared..., kwargs...)
    else  # RenderTransformations — warmup and backend are not forwarded (unused)
        transform_color = isnothing(color) ? true : color
        return render_transformations(
            input;
            npoints,
            color = transform_color,
            alpha,
            resolution,
            outpath,
            ifs_index,
            ifs_name,
            input_fn,
            kwargs...,
        )
    end
end
