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

function _limits_from_points(points::Vector{SVector{2,Float64}})
    return compute_limits(points)
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
    show_divergence_scale::Bool,
    initial_polygon::Union{InitialPolygonPreset,Symbol}=:default,
    backend::Symbol=:cpu
)
    if image_source == :file
        isnothing(image_path) && throw(ArgumentError("image_path is required when image_source=:file"))
        isfile(image_path) || throw(ArgumentError("image_path '$image_path' does not exist"))
        return _to_grayscale_matrix(load(image_path))
    elseif image_source == :chaos
        tifs = IFS(ifs.name, ifs.docs, copy(ifs.points), ifs.maps, ifs.weights, ifs.limits)
        iterate!(tifs; warmup=warmup)
        return make_image(tifs; resolution=resolution, backend=backend)
    elseif image_source == :point_deterministic
        tifs = deterministic_iterate(ifs, deterministic_iters; warmup=warmup)
        return make_image(tifs; resolution=resolution, backend=backend)
    elseif image_source == :inverse
        return rasterize_image_inversely(ifs, inverse_iters, ifs.limits;
                                         resolution=resolution,
                                         show_divergence_scale=show_divergence_scale,
                                         backend=backend)
    elseif image_source == :polygon
        if polygon_limits_mode == :default
            @warn "polygon_limits_mode=:default is treated as :ifs for image_source=:polygon."
        elseif polygon_limits_mode != :ifs
            throw(ArgumentError("Invalid polygon_limits_mode '$polygon_limits_mode'. Supported: :ifs, :default"))
        end
        return _to_grayscale_matrix(_render_transformations_image(ifs;
                                                                   width=resolution[2],
                                                                   height=resolution[1],
                                                                   show_base=false,
                                                                   initial_polygon=initial_polygon,
                                                                   color=false))
    end

    throw(ArgumentError("Invalid image_source '$image_source'. Supported: :polygon, :chaos, :point_deterministic, :inverse, :file"))
end

function _iterate_image_gpu(
    ifs::IFS,
    src::AbstractMatrix{Float32};
    colors::Bool=false,
    seed::Union{Nothing,Integer}=nothing
)
    throw(ArgumentError("GPU backend is not available. Install CUDA.jl and ensure a functional CUDA runtime, or use backend=:cpu/:auto."))
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
    src = _to_grayscale_matrix(img)
    if backend == :cpu
        return _iterate_image_cpu(ifs, src; colors=colors, seed=seed)
    elseif backend == :gpu
        return _iterate_image_gpu(ifs, src; colors=colors, seed=seed)
    elseif backend == :auto
        if _gpu_backend_available(Val(:cuda))
            try
                return _iterate_image_gpu(ifs, src; colors=colors, seed=seed)
            catch err
                if err isa ArgumentError
                    return _iterate_image_cpu(ifs, src; colors=colors, seed=seed)
                end
                rethrow(err)
            end
        end
        return _iterate_image_cpu(ifs, src; colors=colors, seed=seed)
    end
    throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))
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
