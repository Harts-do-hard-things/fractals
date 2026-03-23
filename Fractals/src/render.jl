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

function _resolve_render_iterations(
    iterations::Union{Nothing,Integer},
    deterministic_depth::Integer,
    inverse_depth::Integer
)
    if isnothing(iterations)
        if deterministic_depth != 1 || inverse_depth != 8
            @warn "deterministic_depth/inverse_depth are compatibility aliases; prefer iterations=..."
        end
        deterministic_iters = deterministic_depth
        inverse_iters = inverse_depth
    else
        if deterministic_depth != 1 || inverse_depth != 8
            @warn "iterations takes precedence over deterministic_depth/inverse_depth"
        end
        deterministic_iters = iterations
        inverse_iters = iterations
    end

    deterministic_iters >= 0 || throw(ArgumentError("iterations must be >= 0, got $deterministic_iters"))
    inverse_iters >= 0 || throw(ArgumentError("iterations must be >= 0, got $inverse_iters"))
    return deterministic_iters, inverse_iters
end

function _warn_irrelevant_kwargs(
    method::RenderMethod;
    image_source, image_path, image_iterations,
    polygon_limits_mode, initial_polygon,
    show_divergence_scale,
    color,
)
    if method == ImageIterate && color
        throw(ArgumentError("color=true is not supported for method=ImageIterate. ImageIterate is grayscale-only."))
    end
    if method != ImageIterate
        image_source != :polygon &&
            @warn "image_source=$image_source is ignored for method=$(method); only applies to ImageIterate"
        !isnothing(image_path) &&
            @warn "image_path is ignored for method=$(method); only applies to ImageIterate"
        image_iterations != 1 &&
            @warn "image_iterations=$image_iterations is ignored for method=$(method); only applies to ImageIterate"
        polygon_limits_mode != :ifs &&
            @warn "polygon_limits_mode=$polygon_limits_mode is ignored for method=$(method); only applies to ImageIterate"
        initial_polygon != :default &&
            @warn "initial_polygon=$initial_polygon is ignored for method=$(method); only applies to ImageIterate"
    end
    if method in (Chaos, Parallel, PointDeterministic)
        !show_divergence_scale &&
            @warn "show_divergence_scale=false is ignored for method=$(method); only applies to Inverse"
    end
end

# --------------------------------
# High-level render entrypoint (thin router)
# --------------------------------

function render(
    input;
    method::Union{RenderMethod,Symbol,AbstractString}=Chaos,
    npoints::Union{Nothing,Integer}=nothing,
    warmup::Integer=DEFAULT_WARMUP,
    color::Bool=false,
    show_divergence_scale::Bool=true,
    image_source::Symbol=:polygon,
    image_path::Union{Nothing,AbstractString}=nothing,
    image_iterations::Integer=1,
    polygon_limits_mode::Symbol=:ifs,
    backend::Symbol=:cpu,
    initial_polygon::Symbol=:default,
    resolution::Tuple{Int,Int}=RESOLUTION,
    outpath::AbstractString="media/render.png",
    ifs_index::Union{Nothing,Integer}=nothing,
    ifs_name::Union{Nothing,AbstractString}=nothing,
    iterations::Union{Nothing,Integer}=nothing,
    deterministic_depth::Integer=1,
    inverse_depth::Integer=8,
    input_fn=readline,
)
    parsed_method = _parse_render_method(method)
    render_method = parsed_method == Parallel ? Chaos : parsed_method

    det_iters, inv_iters = _resolve_render_iterations(iterations, deterministic_depth, inverse_depth)

    _warn_irrelevant_kwargs(render_method;
        image_source=image_source, image_path=image_path,
        image_iterations=image_iterations, polygon_limits_mode=polygon_limits_mode,
        initial_polygon=initial_polygon, show_divergence_scale=show_divergence_scale,
        color=color)

    if render_method == Chaos
        return render_chaos(input;
            npoints=npoints, warmup=warmup, color=color,
            resolution=resolution, outpath=outpath, backend=backend,
            ifs_index=ifs_index, ifs_name=ifs_name, input_fn=input_fn)

    elseif render_method == PointDeterministic
        return render_point_deterministic(input;
            npoints=npoints, warmup=warmup, iterations=det_iters, color=color,
            resolution=resolution, outpath=outpath, backend=backend,
            ifs_index=ifs_index, ifs_name=ifs_name, input_fn=input_fn)

    elseif render_method == Inverse
        return render_inverse(input;
            npoints=npoints, warmup=warmup, iterations=inv_iters,
            show_divergence_scale=show_divergence_scale, color=color,
            resolution=resolution, outpath=outpath, backend=backend,
            ifs_index=ifs_index, ifs_name=ifs_name, input_fn=input_fn)

    else  # ImageIterate
        return render_image_iterate(input;
            npoints=npoints, warmup=warmup, iterations=det_iters,
            image_source=image_source, image_path=image_path,
            image_iterations=image_iterations, polygon_limits_mode=polygon_limits_mode,
            initial_polygon=initial_polygon,
            resolution=resolution, outpath=outpath, backend=backend,
            ifs_index=ifs_index, ifs_name=ifs_name, input_fn=input_fn)
    end
end
