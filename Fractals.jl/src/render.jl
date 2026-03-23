# --------------------------------
# Method-specific render functions
# --------------------------------

function render_chaos(
    input;
    npoints::Union{Nothing,Integer}             = nothing,
    warmup::Integer                             = DEFAULT_WARMUP,
    color::Bool                                 = false,
    resolution::Tuple{Int,Int}                  = RESOLUTION,
    outpath::AbstractString                     = "media/render.png",
    backend::Symbol                             = :cpu,
    ifs_index::Union{Nothing,Integer}           = nothing,
    ifs_name::Union{Nothing,AbstractString}     = nothing,
    input_fn                                    = readline,
)
    isnothing(npoints) || npoints > 0 || throw(ArgumentError("npoints must be > 0, got $npoints"))
    warmup >= 0 || throw(ArgumentError("warmup must be >= 0, got $warmup"))
    resolution[1] > 0 && resolution[2] > 0 || throw(ArgumentError("resolution must be positive, got $resolution"))
    backend in (:cpu, :gpu, :auto) || throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))

    ifs = _resolve_render_input(input; npoints=npoints, ifs_index=ifs_index, ifs_name=ifs_name, input_fn=input_fn)
    iterate!(ifs; warmup=warmup)
    img = make_image(ifs; resolution=resolution, backend=backend)

    if color
        img = iterate_image(ifs, img; colors=true, backend=backend)
    end

    final_outpath = _normalize_media_outpath(outpath)
    save(final_outpath, img)
    return (ifs=ifs, image=img, outpath=final_outpath, method=:chaos)
end

function render_point_deterministic(
    input;
    npoints::Union{Nothing,Integer}             = nothing,
    warmup::Integer                             = DEFAULT_WARMUP,
    iterations::Integer                         = 1,
    color::Bool                                 = false,
    resolution::Tuple{Int,Int}                  = RESOLUTION,
    outpath::AbstractString                     = "media/render.png",
    backend::Symbol                             = :cpu,
    ifs_index::Union{Nothing,Integer}           = nothing,
    ifs_name::Union{Nothing,AbstractString}     = nothing,
    input_fn                                    = readline,
)
    isnothing(npoints) || npoints > 0 || throw(ArgumentError("npoints must be > 0, got $npoints"))
    warmup >= 0 || throw(ArgumentError("warmup must be >= 0, got $warmup"))
    iterations >= 0 || throw(ArgumentError("iterations must be >= 0, got $iterations"))
    resolution[1] > 0 && resolution[2] > 0 || throw(ArgumentError("resolution must be positive, got $resolution"))
    backend in (:cpu, :gpu, :auto) || throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))

    ifs = _resolve_render_input(input; npoints=npoints, ifs_index=ifs_index, ifs_name=ifs_name, input_fn=input_fn)
    rendered_ifs = deterministic_iterate(ifs, iterations; warmup=warmup)
    img = make_image(rendered_ifs; resolution=resolution, backend=backend)

    if color
        img = iterate_image(ifs, img; colors=true, backend=backend)
    end

    final_outpath = _normalize_media_outpath(outpath)
    save(final_outpath, img)
    return (ifs=rendered_ifs, image=img, outpath=final_outpath, method=:point_deterministic)
end

function render_inverse(
    input;
    npoints::Union{Nothing,Integer}             = nothing,
    warmup::Integer                             = DEFAULT_WARMUP,
    iterations::Integer                         = 8,
    show_divergence_scale::Bool                 = true,
    color::Bool                                 = false,
    resolution::Tuple{Int,Int}                  = RESOLUTION,
    outpath::AbstractString                     = "media/render.png",
    backend::Symbol                             = :cpu,
    ifs_index::Union{Nothing,Integer}           = nothing,
    ifs_name::Union{Nothing,AbstractString}     = nothing,
    input_fn                                    = readline,
)
    isnothing(npoints) || npoints > 0 || throw(ArgumentError("npoints must be > 0, got $npoints"))
    warmup >= 0 || throw(ArgumentError("warmup must be >= 0, got $warmup"))
    iterations >= 0 || throw(ArgumentError("iterations must be >= 0, got $iterations"))
    resolution[1] > 0 && resolution[2] > 0 || throw(ArgumentError("resolution must be positive, got $resolution"))
    backend in (:cpu, :gpu, :auto) || throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))

    ifs = _resolve_render_input(input; npoints=npoints, ifs_index=ifs_index, ifs_name=ifs_name, input_fn=input_fn)
    img = rasterize_image_inversely(ifs, iterations, ifs.limits;
                                    resolution=resolution,
                                    show_divergence_scale=show_divergence_scale,
                                    backend=backend)

    if color
        img = iterate_image(ifs, img; colors=true, backend=backend)
    end

    final_outpath = _normalize_media_outpath(outpath)
    save(final_outpath, img)
    return (ifs=ifs, image=img, outpath=final_outpath, method=:inverse)
end

function render_image_iterate(
    input;
    npoints::Union{Nothing,Integer}             = nothing,
    warmup::Integer                             = DEFAULT_WARMUP,
    iterations::Integer                         = 1,
    image_source::Symbol                        = :polygon,
    image_path::Union{Nothing,AbstractString}   = nothing,
    image_iterations::Integer                   = 1,
    polygon_limits_mode::Symbol                 = :ifs,
    initial_polygon::Symbol                     = :default,
    resolution::Tuple{Int,Int}                  = RESOLUTION,
    outpath::AbstractString                     = "media/render.png",
    backend::Symbol                             = :cpu,
    ifs_index::Union{Nothing,Integer}           = nothing,
    ifs_name::Union{Nothing,AbstractString}     = nothing,
    # color is intentionally absent — ImageIterate is grayscale-only
    input_fn                                    = readline,
)
    isnothing(npoints) || npoints > 0 || throw(ArgumentError("npoints must be > 0, got $npoints"))
    warmup >= 0 || throw(ArgumentError("warmup must be >= 0, got $warmup"))
    iterations >= 0 || throw(ArgumentError("iterations must be >= 0, got $iterations"))
    image_iterations > 0 || throw(ArgumentError("image_iterations must be > 0, got $image_iterations"))
    resolution[1] > 0 && resolution[2] > 0 || throw(ArgumentError("resolution must be positive, got $resolution"))
    backend in (:cpu, :gpu, :auto) || throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))
    image_source in (:polygon, :chaos, :point_deterministic, :inverse, :file) ||
        throw(ArgumentError("Invalid image_source '$image_source'. Supported: :polygon, :chaos, :point_deterministic, :inverse, :file"))
    polygon_limits_mode in (:ifs, :default) ||
        throw(ArgumentError("Invalid polygon_limits_mode '$polygon_limits_mode'. Supported: :ifs, :default"))
    _resolve_initial_polygon(initial_polygon)
    image_source == :file && isnothing(image_path) &&
        throw(ArgumentError("image_path is required when image_source=:file"))

    ifs = _resolve_render_input(input; npoints=npoints, ifs_index=ifs_index, ifs_name=ifs_name, input_fn=input_fn)
    img = _resolve_image_source(ifs, image_source, image_path, resolution, warmup,
                                iterations, iterations,
                                polygon_limits_mode, true, initial_polygon, backend)
    for _ in 1:image_iterations
        img = iterate_image(ifs, img; colors=false, backend=backend)
    end

    final_outpath = _normalize_media_outpath(outpath)
    save(final_outpath, img)
    return (ifs=ifs, image=img, outpath=final_outpath, method=:image_iterate)
end

function render_transformations(
    input;
    npoints::Union{Nothing,Integer}             = nothing,
    show_base::Bool                             = false,
    axis::Bool                                  = false,
    color::Bool                                 = true,
    initial_polygon::Symbol                     = :default,
    resolution::Tuple{Int,Int}                  = RESOLUTION,
    outpath::AbstractString                     = "media/render.png",
    ifs_index::Union{Nothing,Integer}           = nothing,
    ifs_name::Union{Nothing,AbstractString}     = nothing,
    input_fn                                    = readline,
)
    isnothing(npoints) || npoints > 0 || throw(ArgumentError("npoints must be > 0, got $npoints"))
    resolution[1] > 0 && resolution[2] > 0 || throw(ArgumentError("resolution must be positive, got $resolution"))
    _resolve_initial_polygon(initial_polygon)

    ifs = _resolve_render_input(input; npoints=npoints, ifs_index=ifs_index, ifs_name=ifs_name, input_fn=input_fn)
    img = _render_transformations_image(ifs;
                                        width=resolution[2],
                                        height=resolution[1],
                                        show_base=show_base,
                                        initial_polygon=initial_polygon,
                                        color=color,
                                        axis=axis)
    final_outpath = _normalize_media_outpath(outpath)
    save(final_outpath, img)
    return (ifs=ifs, image=img, outpath=final_outpath, method=:render_transformations)
end

# --------------------------------
# Input resolution helpers
# --------------------------------

function _resolve_render_input(
    input::IFS;
    npoints::Union{Nothing,Integer},
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString},
    input_fn=readline
)
    if !isnothing(ifs_index) || !isnothing(ifs_name)
        throw(ArgumentError("ifs_index/ifs_name are only valid for parsed string/file inputs"))
    end

    if isnothing(npoints) || length(input.points) == npoints
        return input
    end

    return IFS(input.maps, input.weights;
               npoints=npoints,
               name=input.name,
               docs=input.docs,
               limits=input.limits)
end

function _resolve_render_input(
    input::AbstractMatrix{<:Real};
    npoints::Union{Nothing,Integer},
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString},
    input_fn=readline
)
    if !isnothing(ifs_index) || !isnothing(ifs_name)
        throw(ArgumentError("ifs_index/ifs_name are only valid for parsed string/file inputs"))
    end
    resolved_npoints = isnothing(npoints) ? DEFAULT_SAMPLES : npoints
    return IFS(input; npoints=resolved_npoints)
end

function _resolve_render_input(
    input::AbstractString;
    npoints::Union{Nothing,Integer},
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString},
    input_fn=readline  # _select_ifs_definition is defined in interactive.jl
)
    resolved_npoints = isnothing(npoints) ? DEFAULT_SAMPLES : npoints
    defs = isfile(input) ? parse_ifs_definitions_file(input) :
                           parse_ifs_definitions_string(input)

    isempty(defs) && throw(ArgumentError("No IFS definitions found in input"))
    selected = _select_ifs_definition(defs; ifs_index=ifs_index, ifs_name=ifs_name, input_fn=input_fn)
    return IFS(selected.eq; npoints=resolved_npoints, name=selected.name, docs=selected.docs)
end

# --------------------------------
# High-level render entrypoint (thin delegator)
# --------------------------------

function render(
    input;
    method::Union{RenderMethod,Symbol,AbstractString} = Chaos,
    npoints::Union{Nothing,Integer}                   = nothing,
    warmup::Integer                                   = DEFAULT_WARMUP,
    resolution::Tuple{Int,Int}                        = RESOLUTION,
    outpath::AbstractString                           = "media/render.png",
    backend::Symbol                                   = :cpu,
    ifs_index::Union{Nothing,Integer}                 = nothing,
    ifs_name::Union{Nothing,AbstractString}           = nothing,
    input_fn                                          = readline,
    kwargs...
)
    parsed_method = _parse_render_method(method)
    render_method = parsed_method == Parallel ? Chaos : parsed_method

    if render_method == ImageIterate && get(kwargs, :color, false) == true
        throw(ArgumentError("color=true is not supported for method=ImageIterate. ImageIterate is grayscale-only."))
    end

    shared = (; npoints, warmup, resolution, outpath, backend, ifs_index, ifs_name, input_fn)

    if render_method == Chaos
        return render_chaos(input; shared..., kwargs...)
    elseif render_method == PointDeterministic
        return render_point_deterministic(input; shared..., kwargs...)
    elseif render_method == Inverse
        return render_inverse(input; shared..., kwargs...)
    elseif render_method == ImageIterate
        return render_image_iterate(input; shared..., kwargs...)
    else  # RenderTransformations — warmup and backend are not forwarded (unused)
        return render_transformations(input;
                                      npoints, resolution, outpath,
                                      ifs_index, ifs_name, input_fn,
                                      kwargs...)
    end
end
