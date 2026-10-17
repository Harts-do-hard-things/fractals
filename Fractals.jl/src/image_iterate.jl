# NOTE: This file must be included after chaos.jl, point_deterministic.jl, rasterize.jl,
# inverse.jl, and drawing/transformations.jl — _resolve_image_source calls all of them.

# -------------------------------------------------
# Compose iterate map with pixel map
# -------------------------------------------------

function _make_pixeliterate_map(iterate_map::AffineMap{Float64}, pixel_map::AffineMap{Float64})

    inv_pixel = inv(pixel_map)

    A = pixel_map.A * iterate_map.A * inv_pixel.A
    b = pixel_map.A * (iterate_map.A * inv_pixel.b + iterate_map.b) + pixel_map.b

    return AffineMap(A, b)
end

function _normalize_rgb_buffers!(
    rimg::AbstractMatrix{Float32},
    gimg::AbstractMatrix{Float32},
    bimg::AbstractMatrix{Float32},
)
    maxv = max(maximum(rimg), maximum(gimg), maximum(bimg))
    if maxv > 0.0f0
        logmax = log(1.0f0 + maxv)
        @inbounds for i in eachindex(rimg)
            rimg[i] = log(1.0f0 + rimg[i]) / logmax
            gimg[i] = log(1.0f0 + gimg[i]) / logmax
            bimg[i] = log(1.0f0 + bimg[i]) / logmax
        end
    end
    return nothing
end

function _rgb_image_from_buffers(
    rimg::AbstractMatrix{Float32},
    gimg::AbstractMatrix{Float32},
    bimg::AbstractMatrix{Float32},
)
    rows, cols = size(rimg)
    out = Matrix{RGB{Float32}}(undef, rows, cols)
    @inbounds for y = 1:rows
        for x = 1:cols
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
        throw(
            ArgumentError(
                "iterate_image requires a grayscale source image. Convert colored input to Gray first.",
            ),
        )
    end
    return img
end

function _iterate_image_gpu(
    ifs::IFS,
    src::AbstractMatrix{Float32};
    colors::Bool = false,
    seed::Union{Nothing,Integer} = nothing,
)
    throw(_gpu_backend_unavailable_error())
end

function _iterate_image_cpu(
    ifs::IFS,
    src::AbstractMatrix{Float32};
    colors::Bool = false,
    seed::Union{Nothing,Integer} = nothing,
)
    rows, cols = size(src)

    nthreads_local = _thread_buffer_slots()
    buffers = [zeros(Float32, rows, cols) for _ = 1:nthreads_local]
    rbuffers = colors ? [zeros(Float32, rows, cols) for _ = 1:nthreads_local] : nothing
    gbuffers = colors ? [zeros(Float32, rows, cols) for _ = 1:nthreads_local] : nothing
    bbuffers = colors ? [zeros(Float32, rows, cols) for _ = 1:nthreads_local] : nothing
    map_colors = colors ? _map_colors(length(ifs.maps)) : RGB{Float32}[]

    pmap = make_pixelate_map(ifs.limits; resolution = (rows, cols))

    for (map_idx, nmap) in enumerate(ifs.maps)
        cmap = _make_pixeliterate_map(nmap, pmap)  # use it!
        map_color = colors ? _map_color_rgb(map_idx, map_colors) : RGB{Float32}(0.0f0, 0.0f0, 0.0f0)
        cr = map_color.r
        cg = map_color.g
        cb = map_color.b

        @threads for x = 1:cols
            tid = threadid()
            graybuf = buffers[tid]
            rbuf = colors ? rbuffers[tid] : graybuf
            gbuf = colors ? gbuffers[tid] : graybuf
            bbuf = colors ? bbuffers[tid] : graybuf
            @inbounds for y = 1:rows
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
        for t = 2:nthreads_local
            newimg .+= buffers[t]
        end

        maxv = maximum(newimg)
        if maxv > 0.0f0
            logmax = log(1.0f0 + maxv)
            @inbounds for i in eachindex(newimg)
                newimg[i] = log(1.0f0 + newimg[i]) / logmax
            end
        end

        return Gray.(newimg)
    end

    rimg = rbuffers[1]
    gimg = gbuffers[1]
    bimg = bbuffers[1]
    for t = 2:nthreads_local
        rimg .+= rbuffers[t]
        gimg .+= gbuffers[t]
        bimg .+= bbuffers[t]
    end

    _normalize_rgb_buffers!(rimg, gimg, bimg)
    return _rgb_image_from_buffers(rimg, gimg, bimg)
end

"""
    iterate_image(ifs, img; colors=false, seed=nothing, backend=:cpu)

Apply every map of `ifs` to an image in pixel space (framed by `ifs.limits`) and return the
log-normalized sum: one IFS step applied to a picture instead of a point cloud. The input is
converted to grayscale first, so the call can be repeated on its own output to approximate
the attractor.

`colors=false` returns a grayscale `Float32` matrix; `colors=true` tints each map's
contribution and returns `RGB{Float32}`. `backend` behaves as in [`make_image`](@ref).
"""
function iterate_image(
    ifs::IFS,
    img::AbstractMatrix;
    colors::Bool = false,
    seed::Union{Nothing,Integer} = nothing,
    backend::Symbol = :cpu,
)
    _validate_iterate_image_input(img)
    src = _to_grayscale_matrix(img)
    return _dispatch_backend(
        backend,
        () -> _iterate_image_cpu(ifs, src; colors = colors, seed = seed),
        () -> _iterate_image_gpu(ifs, src; colors = colors, seed = seed),
    )
end

function _iterate_image_single_map(ifs::IFS, img::AbstractMatrix{<:Real}, map_index::Integer)
    1 <= map_index <= length(ifs.maps) || throw(
        ArgumentError("map_index $map_index is out of range. Valid range: 1-$(length(ifs.maps))"),
    )
    rows, cols = size(img)
    newimg = zeros(Float32, rows, cols)

    pmap = make_pixelate_map(ifs.limits; resolution = (rows, cols))
    nmap = ifs.maps[map_index]
    cmap = _make_pixeliterate_map(nmap, pmap)

    @inbounds for x = 1:cols
        for y = 1:rows
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
    if maxv > 0.0f0
        logmax = log(1.0f0 + maxv)
        @inbounds for i in eachindex(newimg)
            newimg[i] = log(1.0f0 + newimg[i]) / logmax
        end
    end

    return Gray.(newimg)
end
