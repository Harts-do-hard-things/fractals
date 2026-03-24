# NOTE: This file must be included after chaos.jl, point_deterministic.jl, rasterize.jl,
# inverse.jl, and render_transformations.jl — _resolve_image_source calls all of them.

# -------------------------------------------------
# Compose iterate map with pixel map
# -------------------------------------------------

function _make_pixeliterate_map(
    iterate_map::AffineMap{Float64},
    pixel_map::AffineMap{Float64}
)

    inv_pixel = inv(pixel_map)

    A = pixel_map.A * iterate_map.A * inv_pixel.A
    b = pixel_map.A * (iterate_map.A * inv_pixel.b + iterate_map.b) +
        pixel_map.b

    return AffineMap(A, b)
end

function _normalize_rgb_buffers!(
    rimg::AbstractMatrix{Float32},
    gimg::AbstractMatrix{Float32},
    bimg::AbstractMatrix{Float32}
)
    maxv = max(maximum(rimg), maximum(gimg), maximum(bimg))
    if maxv > 0f0
        logmax = log(1f0 + maxv)
        @inbounds for i in eachindex(rimg)
            rimg[i] = log(1f0 + rimg[i]) / logmax
            gimg[i] = log(1f0 + gimg[i]) / logmax
            bimg[i] = log(1f0 + bimg[i]) / logmax
        end
    end
    return nothing
end

function _rgb_image_from_buffers(
    rimg::AbstractMatrix{Float32},
    gimg::AbstractMatrix{Float32},
    bimg::AbstractMatrix{Float32}
)
    rows, cols = size(rimg)
    out = Matrix{RGB{Float32}}(undef, rows, cols)
    @inbounds for y in 1:rows
        for x in 1:cols
            out[y, x] = RGB{Float32}(rimg[y, x], gimg[y, x], bimg[y, x])
        end
    end
    return out
end

function _to_grayscale_matrix(img::AbstractMatrix)
    return Float32.(Gray.(img))
end

function _validate_iterate_image_input(img::AbstractMatrix)
    if eltype(img) <: Colorant && !(color_type(eltype(img)) <: Gray)
        throw(ArgumentError("iterate_image requires a grayscale source image. Convert colored input to Gray first."))
    end
    return img
end

function _limits_from_points(points::Vector{SVector{2,Float64}})
    return compute_limits(points)
end

@inline function _normalize_polygon_limits_mode(polygon_limits_mode::Symbol)
    if polygon_limits_mode == :default
        @warn "polygon_limits_mode=:default is treated as :geometry for image_source=:polygon."
        return :geometry
    elseif polygon_limits_mode in (:ifs, :geometry)
        return polygon_limits_mode
    end
    throw(ArgumentError("Invalid polygon_limits_mode '$polygon_limits_mode'. Supported: :geometry, :ifs, :default"))
end

struct ImageSourceRequest
    ifs::IFS
    image_source::Symbol
    image_path::Union{Nothing,AbstractString}
    resolution::Tuple{Int,Int}
    warmup::Int
    deterministic_iters::Int
    inverse_iters::Int
    polygon_limits_mode::Symbol
    polygon_limits_iterations::Int
    show_divergence_scale::Bool
    initial_polygon_spec::InitialPolygonPreset
    backend::Symbol
end

const _IMAGE_SOURCE_CHOICES = (:polygon, :chaos, :point_deterministic, :inverse, :file)

@inline function _image_source_names_text()
    return join(":" .* string.(_IMAGE_SOURCE_CHOICES), ", ")
end

@inline function _validate_image_source(image_source::Symbol)
    image_source in _IMAGE_SOURCE_CHOICES ||
        throw(ArgumentError("Invalid image_source '$image_source'. Supported: $(_image_source_names_text())"))
    return image_source
end

function _resolve_image_source_from_file(req::ImageSourceRequest)
    isnothing(req.image_path) && throw(ArgumentError("image_path is required when image_source=:file"))
    isfile(req.image_path) || throw(ArgumentError("image_path '$(req.image_path)' does not exist"))
    img = load(req.image_path)
    _validate_iterate_image_input(img)
    source_limits = _read_png_source_limits(req.image_path)
    source_ifs = isnothing(source_limits) ? req.ifs : _ifs_with_limits(req.ifs, source_limits)
    return (; ifs=source_ifs, image=_to_grayscale_matrix(img))
end

function _resolve_image_source_from_chaos(req::ImageSourceRequest)
    tifs = IFS(req.ifs.name, req.ifs.docs, copy(req.ifs.points), req.ifs.maps, req.ifs.weights, req.ifs.limits)
    iterate!(tifs; warmup=req.warmup)
    return (; ifs=req.ifs, image=make_image(tifs; resolution=req.resolution, backend=req.backend))
end

function _resolve_image_source_from_point_deterministic(req::ImageSourceRequest)
    tifs = deterministic_iterate(req.ifs, req.deterministic_iters; warmup=req.warmup)
    return (; ifs=req.ifs, image=make_image(tifs; resolution=req.resolution, backend=req.backend))
end

function _resolve_image_source_from_inverse(req::ImageSourceRequest)
    return (; ifs=req.ifs,
            image=rasterize_image_inversely(req.ifs, req.inverse_iters, req.ifs.limits;
                                            resolution=req.resolution,
                                            show_divergence_scale=req.show_divergence_scale,
                                            backend=req.backend))
end

function _resolve_image_source_from_polygon(req::ImageSourceRequest)
    limits_mode = _normalize_polygon_limits_mode(req.polygon_limits_mode)
    raster_ifs = limits_mode == :geometry ?
        _build_polygon_raster_ifs(req.ifs;
                                  initial_polygon_spec=req.initial_polygon_spec,
                                  polygon_limits_iterations=req.polygon_limits_iterations) :
        req.ifs
    seed = _render_initial_polygon_seed_image(raster_ifs;
                                              width=req.resolution[2],
                                              height=req.resolution[1],
                                              initial_polygon_spec=req.initial_polygon_spec)
    return (; ifs=raster_ifs, image=_to_grayscale_matrix(seed))
end

function _occupied_bbox(img::AbstractMatrix)
    xmin = typemax(Int)
    xmax = typemin(Int)
    ymin = typemax(Int)
    ymax = typemin(Int)
    found = false
    @inbounds for y in axes(img, 1), x in axes(img, 2)
        if Float32(gray(img[y, x])) > 0f0
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            found = true
        end
    end
    return found ? (xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax) : nothing
end

const _IMAGE_SOURCE_DISPATCH = Dict{Symbol,Function}(
    :file => _resolve_image_source_from_file,
    :chaos => _resolve_image_source_from_chaos,
    :point_deterministic => _resolve_image_source_from_point_deterministic,
    :inverse => _resolve_image_source_from_inverse,
    :polygon => _resolve_image_source_from_polygon,
)

function _resolve_image_source_context(req::ImageSourceRequest)
    handler = get(_IMAGE_SOURCE_DISPATCH, _validate_image_source(req.image_source), nothing)
    handler === nothing &&
        throw(ArgumentError("Invalid image_source '$(req.image_source)'. Supported: $(_image_source_names_text())"))
    return handler(req)
end

function _resolve_image_source_context(
    ifs::IFS,
    image_source::Symbol,
    image_path::Union{Nothing,AbstractString},
    resolution::Tuple{Int,Int},
    warmup::Integer,
    deterministic_iters::Integer,
    inverse_iters::Integer,
    polygon_limits_mode::Symbol,
    polygon_limits_iterations::Integer,
    show_divergence_scale::Bool,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol}=initial_polygon(),
    backend::Symbol=:cpu
)
    return _resolve_image_source_context(ImageSourceRequest(
        ifs,
        image_source,
        image_path,
        resolution,
        Int(warmup),
        Int(deterministic_iters),
        Int(inverse_iters),
        polygon_limits_mode,
        Int(polygon_limits_iterations),
        show_divergence_scale,
        _resolve_initial_polygon(initial_polygon_spec),
        backend,
    ))
end

function _resolve_image_source(
    ifs::IFS,
    image_source::Symbol,
    image_path::Union{Nothing,AbstractString},
    resolution::Tuple{Int,Int},
    warmup::Integer,
    deterministic_iters::Integer,
    inverse_iters::Integer,
    polygon_limits_mode::Symbol,
    polygon_limits_iterations::Integer,
    show_divergence_scale::Bool,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol}=initial_polygon(),
    backend::Symbol=:cpu
)::AbstractMatrix
    return _resolve_image_source_context(ifs,
                                         image_source,
                                         image_path,
                                         resolution,
                                         warmup,
                                         deterministic_iters,
                                         inverse_iters,
                                         polygon_limits_mode,
                                         polygon_limits_iterations,
                                         show_divergence_scale,
                                         initial_polygon_spec,
                                         backend).image
end

function _iterate_image_gpu(
    ifs::IFS,
    src::AbstractMatrix{Float32};
    colors::Bool=false,
    seed::Union{Nothing,Integer}=nothing
)
    throw(_gpu_backend_unavailable_error())
end

function _iterate_image_cpu(
    ifs::IFS,
    src::AbstractMatrix{Float32};
    colors::Bool=false,
    seed::Union{Nothing,Integer}=nothing
)
    rows, cols = size(src)

    nthreads_local = _thread_buffer_slots()
    buffers = [zeros(Float32, rows, cols) for _ in 1:nthreads_local]
    rbuffers = colors ? [zeros(Float32, rows, cols) for _ in 1:nthreads_local] : nothing
    gbuffers = colors ? [zeros(Float32, rows, cols) for _ in 1:nthreads_local] : nothing
    bbuffers = colors ? [zeros(Float32, rows, cols) for _ in 1:nthreads_local] : nothing
    map_colors = colors ? _map_colors(length(ifs.maps)) : RGB{Float32}[]

    pmap = make_pixelate_map(ifs.limits; resolution=(rows, cols))

    for (map_idx, nmap) in enumerate(ifs.maps)
        cmap = _make_pixeliterate_map(nmap, pmap)  # use it!
        map_color = colors ? _map_color_rgb(map_idx, map_colors) : RGB{Float32}(0.0f0, 0.0f0, 0.0f0)
        cr = map_color.r
        cg = map_color.g
        cb = map_color.b

        @threads for x in 1:cols
            tid = threadid()
            graybuf = buffers[tid]
            rbuf = colors ? rbuffers[tid] : graybuf
            gbuf = colors ? gbuffers[tid] : graybuf
            bbuf = colors ? bbuffers[tid] : graybuf
            @inbounds for y in 1:rows
                val = src[y, x]
                if val != 0
                    fx, fy = cmap(SVector(x, y))

                    newx = round(Int, fx)
                    newy = round(Int, fy)
                    if !(1 <= newx <= cols && 1 <= newy <= rows)
                        continue
                    end

                    if colors
                        fval = Float32(val)
                        rbuf[newy, newx] += fval * cr
                        gbuf[newy, newx] += fval * cg
                        bbuf[newy, newx] += fval * cb
                    else
                        graybuf[newy, newx] += val
                    end
                end
            end
        end
    end

    if !colors
        newimg = buffers[1]
        for t in 2:nthreads_local
            newimg .+= buffers[t]
        end

        maxv = maximum(newimg)
        if maxv > 0f0
            logmax = log(1f0 + maxv)
            @inbounds for i in eachindex(newimg)
                newimg[i] = log(1f0 + newimg[i]) / logmax
            end
        end

        return Gray.(newimg)
    end

    rimg = rbuffers[1]
    gimg = gbuffers[1]
    bimg = bbuffers[1]
    for t in 2:nthreads_local
        rimg .+= rbuffers[t]
        gimg .+= gbuffers[t]
        bimg .+= bbuffers[t]
    end

    _normalize_rgb_buffers!(rimg, gimg, bimg)
    return _rgb_image_from_buffers(rimg, gimg, bimg)
end

function iterate_image(
    ifs::IFS,
    img::AbstractMatrix;
    colors::Bool=false,
    seed::Union{Nothing,Integer}=nothing,
    backend::Symbol=:cpu
)
    _validate_iterate_image_input(img)
    src = _to_grayscale_matrix(img)
    return _dispatch_backend(
        backend,
        () -> _iterate_image_cpu(ifs, src; colors=colors, seed=seed),
        () -> _iterate_image_gpu(ifs, src; colors=colors, seed=seed),
    )
end

function _iterate_image_single_map(ifs::IFS, img::AbstractMatrix{<:Real}, map_index::Integer)
    1 <= map_index <= length(ifs.maps) || throw(ArgumentError("map_index $map_index is out of range. Valid range: 1-$(length(ifs.maps))"))
    rows, cols = size(img)
    newimg = zeros(Float32, rows, cols)

    pmap = make_pixelate_map(ifs.limits; resolution=(rows, cols))
    nmap = ifs.maps[map_index]
    cmap = _make_pixeliterate_map(nmap, pmap)

    @inbounds for x in 1:cols
        for y in 1:rows
            val = img[y, x]
            if val != 0
                fx, fy = cmap(SVector(x, y))
                newx = round(Int, fx)
                newy = round(Int, fy)
                if !(1 <= newx <= cols && 1 <= newy <= rows)
                    continue
                end
                newimg[newy, newx] += val
            end
        end
    end

    maxv = maximum(newimg)
    if maxv > 0f0
        logmax = log(1f0 + maxv)
        @inbounds for i in eachindex(newimg)
            newimg[i] = log(1f0 + newimg[i]) / logmax
        end
    end

    return Gray.(newimg)
end
