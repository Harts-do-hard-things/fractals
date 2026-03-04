module FractalsCUDAExt

using CUDA
using Fractals

import Fractals: IFS, RESOLUTION, _gpu_backend_available, _make_image_gpu, _rasterize_image_inversely_gpu, _iterate_image_gpu

@inline function _gpu_backend_available(::Val{:cuda})
    try
        return CUDA.functional()
    catch
        return false
    end
end

@inline function _in_limits(x::Float64, y::Float64, xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64)
    return (xmin <= x <= xmax) && (ymin <= y <= ymax)
end

@inline function _tri_has_any_corner_in_limits(
    p0x::Float64, p0y::Float64,
    p1x::Float64, p1y::Float64,
    p2x::Float64, p2y::Float64,
    xmin::Float64, xmax::Float64,
    ymin::Float64, ymax::Float64
)
    return _in_limits(p0x, p0y, xmin, xmax, ymin, ymax) ||
           _in_limits(p1x, p1y, xmin, xmax, ymin, ymax) ||
           _in_limits(p2x, p2y, xmin, xmax, ymin, ymax)
end

@inline function _points_contain_zero_gpu(
    p0x::Float64, p0y::Float64,
    p1x::Float64, p1y::Float64,
    p2x::Float64, p2y::Float64
)
    v1x = p1x - p0x
    v1y = p1y - p0y
    v2x = p2x - p0x
    v2y = p2y - p0y
    vox = -p0x
    voy = -p0y

    d11 = v1x * v1x + v1y * v1y
    d22 = v2x * v2x + v2y * v2y
    d1o = v1x * vox + v1y * voy
    d2o = v2x * vox + v2y * voy
    return (0.0 <= d1o <= d11) && (0.0 <= d2o <= d22)
end

@inline function _inverse_value_for_display(v::Float32, show_divergence_scale::Bool)
    if show_divergence_scale
        return v
    end
    return (v == 0.0f0 || v == 1.0f0) ? 1.0f0 : 0.0f0
end

function _inverse_rasterize_kernel!(
    img,
    overflow,
    cur_p0x, cur_p0y, cur_p1x, cur_p1y, cur_p2x, cur_p2y,
    nxt_p0x, nxt_p0y, nxt_p1x, nxt_p1y, nxt_p2x, nxt_p2y,
    ia11::Float64, ia12::Float64, ia21::Float64, ia22::Float64, ib1::Float64, ib2::Float64,
    map_a11, map_a12, map_a21, map_a22, map_b1, map_b2,
    n::Int32, nmaps::Int32, cap::Int32,
    rows::Int32, npix::Int32,
    xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64,
    show_divergence_scale::Bool,
    exact_mode::Bool
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > npix
        return
    end

    row = ((idx - 1) % rows) + 1
    col = ((idx - 1) ÷ rows) + 1

    x = Float64(col)
    y = Float64(row)

    p0x = ia11 * x + ia12 * y + ib1
    p0y = ia21 * x + ia22 * y + ib2
    p1x = ia11 * (x + 1.0) + ia12 * y + ib1
    p1y = ia21 * (x + 1.0) + ia22 * y + ib2
    p2x = ia11 * x + ia12 * (y + 1.0) + ib1
    p2y = ia21 * x + ia22 * (y + 1.0) + ib2

    cur_count = Int32(1)
    cur_p0x[1, idx] = p0x
    cur_p0y[1, idx] = p0y
    cur_p1x[1, idx] = p1x
    cur_p1y[1, idx] = p1y
    cur_p2x[1, idx] = p2x
    cur_p2y[1, idx] = p2y
    use_cur = true

    j = Int32(1)
    while j <= n
        next_count = Int32(0)

        if use_cur
            t = Int32(1)
            while t <= cur_count
                c0x = cur_p0x[t, idx]; c0y = cur_p0y[t, idx]
                c1x = cur_p1x[t, idx]; c1y = cur_p1y[t, idx]
                c2x = cur_p2x[t, idx]; c2y = cur_p2y[t, idx]

                if _tri_has_any_corner_in_limits(c0x, c0y, c1x, c1y, c2x, c2y, xmin, xmax, ymin, ymax)
                    m = Int32(1)
                    while m <= nmaps
                        if next_count < cap
                            a11 = map_a11[m]; a12 = map_a12[m]
                            a21 = map_a21[m]; a22 = map_a22[m]
                            b1 = map_b1[m]; b2 = map_b2[m]
                            next_count += 1
                            nxt_p0x[next_count, idx] = a11 * c0x + a12 * c0y + b1
                            nxt_p0y[next_count, idx] = a21 * c0x + a22 * c0y + b2
                            nxt_p1x[next_count, idx] = a11 * c1x + a12 * c1y + b1
                            nxt_p1y[next_count, idx] = a21 * c1x + a22 * c1y + b2
                            nxt_p2x[next_count, idx] = a11 * c2x + a12 * c2y + b1
                            nxt_p2y[next_count, idx] = a21 * c2x + a22 * c2y + b2
                        elseif exact_mode
                            overflow[idx] = 1
                            img[idx] = _inverse_value_for_display(0.0f0, show_divergence_scale)
                            return
                        end
                        m += 1
                    end
                end
                t += 1
            end
        else
            t = Int32(1)
            while t <= cur_count
                c0x = nxt_p0x[t, idx]; c0y = nxt_p0y[t, idx]
                c1x = nxt_p1x[t, idx]; c1y = nxt_p1y[t, idx]
                c2x = nxt_p2x[t, idx]; c2y = nxt_p2y[t, idx]

                if _tri_has_any_corner_in_limits(c0x, c0y, c1x, c1y, c2x, c2y, xmin, xmax, ymin, ymax)
                    m = Int32(1)
                    while m <= nmaps
                        if next_count < cap
                            a11 = map_a11[m]; a12 = map_a12[m]
                            a21 = map_a21[m]; a22 = map_a22[m]
                            b1 = map_b1[m]; b2 = map_b2[m]
                            next_count += 1
                            cur_p0x[next_count, idx] = a11 * c0x + a12 * c0y + b1
                            cur_p0y[next_count, idx] = a21 * c0x + a22 * c0y + b2
                            cur_p1x[next_count, idx] = a11 * c1x + a12 * c1y + b1
                            cur_p1y[next_count, idx] = a21 * c1x + a22 * c1y + b2
                            cur_p2x[next_count, idx] = a11 * c2x + a12 * c2y + b1
                            cur_p2y[next_count, idx] = a21 * c2x + a22 * c2y + b2
                        elseif exact_mode
                            overflow[idx] = 1
                            img[idx] = _inverse_value_for_display(0.0f0, show_divergence_scale)
                            return
                        end
                        m += 1
                    end
                end
                t += 1
            end
        end

        if next_count == 0
            v = Float32(j) / Float32(max(n, 1)) * 0.5f0
            img[idx] = _inverse_value_for_display(v, show_divergence_scale)
            return
        end

        contains_zero = false
        if use_cur
            t = Int32(1)
            while t <= next_count
                if _points_contain_zero_gpu(
                    nxt_p0x[t, idx], nxt_p0y[t, idx],
                    nxt_p1x[t, idx], nxt_p1y[t, idx],
                    nxt_p2x[t, idx], nxt_p2y[t, idx]
                )
                    contains_zero = true
                    break
                end
                t += 1
            end
        else
            t = Int32(1)
            while t <= next_count
                if _points_contain_zero_gpu(
                    cur_p0x[t, idx], cur_p0y[t, idx],
                    cur_p1x[t, idx], cur_p1y[t, idx],
                    cur_p2x[t, idx], cur_p2y[t, idx]
                )
                    contains_zero = true
                    break
                end
                t += 1
            end
        end

        if contains_zero
            img[idx] = _inverse_value_for_display(1.0f0, show_divergence_scale)
            return
        end

        cur_count = next_count
        use_cur = !use_cur
        j += 1
    end

    img[idx] = _inverse_value_for_display(0.0f0, show_divergence_scale)
    return
end

function _rasterize_points_kernel!(
    img,
    xs,
    ys,
    n::Int32,
    nplanes::Int32,
    rows::Int32,
    cols::Int32,
    sx::Float32,
    sy::Float32,
    bx::Float32,
    by::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        fx = sx * xs[i] + bx
        fy = sy * ys[i] + by
        px = Int(round(fx))
        py = Int(round(fy))
        if px < 1
            px = 1
        elseif px > cols
            px = cols
        end
        if py < 1
            py = 1
        elseif py > rows
            py = rows
        end
        plane = ((i - 1) % nplanes) + 1
        CUDA.@atomic img[py, px, plane] += 1.0f0
        i += stride
    end
    return
end

function _reduce_planes_kernel!(
    dst,
    src,
    nplanes::Int32,
    rows::Int32,
    npix::Int32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > npix
        return
    end

    y = ((i - 1) % rows) + 1
    x = ((i - 1) ÷ rows) + 1
    acc = 0.0f0
    p = Int32(1)
    while p <= nplanes
        acc += src[y, x, p]
        p += 1
    end
    dst[y, x] = acc
    return
end

function _normalize_image_kernel!(img, n::Int32, logmax::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds img[i] = log1p(img[i]) / logmax
    end
    return
end

function _iterate_image_gray_kernel!(
    dst,
    src,
    map_a11,
    map_a12,
    map_a21,
    map_a22,
    map_b1,
    map_b2,
    nmaps::Int32,
    rows::Int32,
    cols::Int32,
    npix::Int32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > npix
        return
    end

    y = ((i - 1) % rows) + 1
    x = ((i - 1) ÷ rows) + 1
    @inbounds val = src[y, x]
    if val == 0f0
        return
    end

    fx0 = Float64(x)
    fy0 = Float64(y)
    m = Int32(1)
    while m <= nmaps
        fx = map_a11[m] * fx0 + map_a12[m] * fy0 + map_b1[m]
        fy = map_a21[m] * fx0 + map_a22[m] * fy0 + map_b2[m]

        px = Int(round(fx))
        py = Int(round(fy))
        if 1 <= px <= cols && 1 <= py <= rows
            CUDA.@atomic dst[py, px] += val
        end
        m += 1
    end
    return
end

function _iterate_image_rgb_kernel!(
    rdst,
    gdst,
    bdst,
    src,
    map_a11,
    map_a12,
    map_a21,
    map_a22,
    map_b1,
    map_b2,
    map_cr,
    map_cg,
    map_cb,
    nmaps::Int32,
    rows::Int32,
    cols::Int32,
    npix::Int32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > npix
        return
    end

    y = ((i - 1) % rows) + 1
    x = ((i - 1) ÷ rows) + 1
    @inbounds val = src[y, x]
    if val == 0f0
        return
    end

    fx0 = Float64(x)
    fy0 = Float64(y)
    m = Int32(1)
    while m <= nmaps
        fx = map_a11[m] * fx0 + map_a12[m] * fy0 + map_b1[m]
        fy = map_a21[m] * fx0 + map_a22[m] * fy0 + map_b2[m]

        px = Int(round(fx))
        py = Int(round(fy))
        if 1 <= px <= cols && 1 <= py <= rows
            CUDA.@atomic rdst[py, px] += val * map_cr[m]
            CUDA.@atomic gdst[py, px] += val * map_cg[m]
            CUDA.@atomic bdst[py, px] += val * map_cb[m]
        end
        m += 1
    end
    return
end

function _normalize_rgb_image_kernel!(rimg, gimg, bimg, n::Int32, logmax::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds rimg[i] = log1p(rimg[i]) / logmax
        @inbounds gimg[i] = log1p(gimg[i]) / logmax
        @inbounds bimg[i] = log1p(bimg[i]) / logmax
    end
    return
end

@inline function _launch_threads(kernel, n::Integer)::Int
    cfg = CUDA.launch_configuration(kernel.fun)
    n <= 0 && return 1
    return max(32, min(Int(cfg.threads), Int(n), 512))
end

@inline function _make_image_planes(npts::Int, rows::Int, cols::Int)::Int
    density = npts / max(rows * cols, 1)
    if npts >= 500_000 || density >= 8
        return 4
    elseif npts >= 100_000 || density >= 2
        return 2
    end
    return 1
end

function _make_image_gpu(ifs::IFS, ::Val{:cuda}; resolution::Tuple{Int,Int}=RESOLUTION)
    _gpu_backend_available(Val(:cuda)) ||
        throw(ArgumentError("GPU backend is not available. Ensure CUDA.jl is installed and CUDA.functional() is true."))

    rows, cols = resolution
    (xlim, ylim) = ifs.limits
    xmin, xmax = xlim
    ymin, ymax = ylim

    r = min(rows, cols)
    sx = Float32(r / (xmax - xmin))
    sy = Float32(r / (ymax - ymin))
    bx = Float32(-xmin * sx + (cols - r) / 2 + 0.5)
    by = Float32(-ymin * sy + (rows - r) / 2 + 0.5)

    npts = length(ifs.points)
    xs_h = Vector{Float32}(undef, npts)
    ys_h = Vector{Float32}(undef, npts)
    @inbounds for i in 1:npts
        p = ifs.points[i]
        xs_h[i] = Float32(p[1])
        ys_h[i] = Float32(p[2])
    end

    xs = CuArray(xs_h)
    ys = CuArray(ys_h)
    nplanes = _make_image_planes(npts, rows, cols)
    # Keep temporary multi-plane buffer bounded and fall back to one plane if too large.
    bytes = rows * cols * nplanes * sizeof(Float32)
    max_bytes = 256 * 1024 * 1024
    if bytes > max_bytes
        nplanes = 1
    end

    img_planes_d = CUDA.zeros(Float32, rows, cols, nplanes)
    img_d = CUDA.zeros(Float32, rows, cols)

    raster_kernel = @cuda launch=false _rasterize_points_kernel!(
        img_planes_d,
        xs,
        ys,
        Int32(npts),
        Int32(nplanes),
        Int32(rows),
        Int32(cols),
        sx,
        sy,
        bx,
        by
    )
    threads_points = _launch_threads(raster_kernel, npts)
    blocks_points = cld(npts, threads_points)
    raster_kernel(
        img_planes_d,
        xs,
        ys,
        Int32(npts),
        Int32(nplanes),
        Int32(rows),
        Int32(cols),
        sx,
        sy,
        bx,
        by;
        threads=threads_points,
        blocks=blocks_points
    )

    npix = rows * cols
    reduce_kernel = @cuda launch=false _reduce_planes_kernel!(
        img_d,
        img_planes_d,
        Int32(nplanes),
        Int32(rows),
        Int32(npix)
    )
    threads_reduce = _launch_threads(reduce_kernel, npix)
    blocks_reduce = cld(npix, threads_reduce)
    reduce_kernel(
        img_d,
        img_planes_d,
        Int32(nplanes),
        Int32(rows),
        Int32(npix);
        threads=threads_reduce,
        blocks=blocks_reduce
    )

    maxv = CUDA.maximum(img_d)
    if maxv > 0f0
        logmax = log1p(maxv)
        normalize_kernel = @cuda launch=false _normalize_image_kernel!(img_d, Int32(npix), logmax)
        threads_norm = _launch_threads(normalize_kernel, npix)
        blocks_norm = cld(npix, threads_norm)
        normalize_kernel(img_d, Int32(npix), logmax; threads=threads_norm, blocks=blocks_norm)
    end

    return Array(img_d)
end

function _rasterize_image_inversely_gpu(
    ifs::IFS,
    n::Integer,
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}},
    ::Val{:cuda};
    resolution::Tuple{Int,Int}=RESOLUTION,
    show_divergence_scale::Bool=true,
    mode::Symbol=:exact
)
    _gpu_backend_available(Val(:cuda)) ||
        throw(ArgumentError("GPU backend is not available. Ensure CUDA.jl is installed and CUDA.functional() is true."))

    mode in (:exact, :preview) ||
        throw(ArgumentError("Invalid mode '$mode'. Supported: :exact, :preview"))
    n >= 0 || throw(ArgumentError("n must be >= 0, got $n"))

    cap = Fractals._inverse_gpu_capacity(ifs, n, resolution, mode)
    rows, cols = resolution
    npix = rows * cols

    inv_pixel_map = inv(Fractals.make_pixelate_map(limits; resolution=resolution))
    ia11 = Float64(inv_pixel_map.A[1, 1]); ia12 = Float64(inv_pixel_map.A[1, 2])
    ia21 = Float64(inv_pixel_map.A[2, 1]); ia22 = Float64(inv_pixel_map.A[2, 2])
    ib1 = Float64(inv_pixel_map.b[1]); ib2 = Float64(inv_pixel_map.b[2])

    inverse_maps = inv.(ifs.maps)
    nmaps = length(inverse_maps)

    map_a11_h = Vector{Float64}(undef, nmaps)
    map_a12_h = Vector{Float64}(undef, nmaps)
    map_a21_h = Vector{Float64}(undef, nmaps)
    map_a22_h = Vector{Float64}(undef, nmaps)
    map_b1_h = Vector{Float64}(undef, nmaps)
    map_b2_h = Vector{Float64}(undef, nmaps)
    @inbounds for i in 1:nmaps
        m = inverse_maps[i]
        map_a11_h[i] = m.A[1, 1]
        map_a12_h[i] = m.A[1, 2]
        map_a21_h[i] = m.A[2, 1]
        map_a22_h[i] = m.A[2, 2]
        map_b1_h[i] = m.b[1]
        map_b2_h[i] = m.b[2]
    end

    map_a11 = CuArray(map_a11_h); map_a12 = CuArray(map_a12_h)
    map_a21 = CuArray(map_a21_h); map_a22 = CuArray(map_a22_h)
    map_b1 = CuArray(map_b1_h); map_b2 = CuArray(map_b2_h)

    cur_p0x = CuArray{Float64}(undef, cap, npix)
    cur_p0y = CuArray{Float64}(undef, cap, npix)
    cur_p1x = CuArray{Float64}(undef, cap, npix)
    cur_p1y = CuArray{Float64}(undef, cap, npix)
    cur_p2x = CuArray{Float64}(undef, cap, npix)
    cur_p2y = CuArray{Float64}(undef, cap, npix)
    nxt_p0x = CuArray{Float64}(undef, cap, npix)
    nxt_p0y = CuArray{Float64}(undef, cap, npix)
    nxt_p1x = CuArray{Float64}(undef, cap, npix)
    nxt_p1y = CuArray{Float64}(undef, cap, npix)
    nxt_p2x = CuArray{Float64}(undef, cap, npix)
    nxt_p2y = CuArray{Float64}(undef, cap, npix)

    img_d = CUDA.zeros(Float32, rows, cols)
    img_vec = reshape(img_d, :)
    overflow = CUDA.zeros(UInt8, npix)

    xmin, xmax = limits[1]
    ymin, ymax = limits[2]

    threads = 256
    blocks = cld(npix, threads)
    @cuda threads=threads blocks=blocks _inverse_rasterize_kernel!(
        img_vec,
        overflow,
        cur_p0x, cur_p0y, cur_p1x, cur_p1y, cur_p2x, cur_p2y,
        nxt_p0x, nxt_p0y, nxt_p1x, nxt_p1y, nxt_p2x, nxt_p2y,
        ia11, ia12, ia21, ia22, ib1, ib2,
        map_a11, map_a12, map_a21, map_a22, map_b1, map_b2,
        Int32(n), Int32(nmaps), Int32(cap),
        Int32(rows), Int32(npix),
        Float64(xmin), Float64(xmax), Float64(ymin), Float64(ymax),
        show_divergence_scale,
        mode == :exact
    )

    if mode == :exact
        overflow_h = Array(overflow)
        if any(!iszero, overflow_h)
            throw(ArgumentError("Inverse GPU exact mode overflowed internal triangle capacity; reduce iterations/resolution or use backend=:cpu."))
        end
    end

    return Array(img_d)
end

function _iterate_image_gpu(
    ifs::IFS,
    src::AbstractMatrix{Float32};
    colors::Bool=false,
    seed::Union{Nothing,Integer}=nothing
)
    _gpu_backend_available(Val(:cuda)) ||
        throw(ArgumentError("GPU backend is not available. Ensure CUDA.jl is installed and CUDA.functional() is true."))

    _ = seed
    rows, cols = size(src)
    npix = rows * cols
    nmaps = length(ifs.maps)

    pmap = Fractals.make_pixelate_map(ifs.limits; resolution=(rows, cols))
    map_a11_h = Vector{Float64}(undef, nmaps)
    map_a12_h = Vector{Float64}(undef, nmaps)
    map_a21_h = Vector{Float64}(undef, nmaps)
    map_a22_h = Vector{Float64}(undef, nmaps)
    map_b1_h = Vector{Float64}(undef, nmaps)
    map_b2_h = Vector{Float64}(undef, nmaps)
    @inbounds for i in 1:nmaps
        cmap = Fractals._make_pixeliterate_map(ifs.maps[i], pmap)
        map_a11_h[i] = cmap.A[1, 1]
        map_a12_h[i] = cmap.A[1, 2]
        map_a21_h[i] = cmap.A[2, 1]
        map_a22_h[i] = cmap.A[2, 2]
        map_b1_h[i] = cmap.b[1]
        map_b2_h[i] = cmap.b[2]
    end

    map_a11 = CuArray(map_a11_h)
    map_a12 = CuArray(map_a12_h)
    map_a21 = CuArray(map_a21_h)
    map_a22 = CuArray(map_a22_h)
    map_b1 = CuArray(map_b1_h)
    map_b2 = CuArray(map_b2_h)

    src_d = CuArray(src)
    threads = 256
    blocks = cld(npix, threads)

    if !colors
        out_d = CUDA.zeros(Float32, rows, cols)
        @cuda threads=threads blocks=blocks _iterate_image_gray_kernel!(
            out_d,
            src_d,
            map_a11,
            map_a12,
            map_a21,
            map_a22,
            map_b1,
            map_b2,
            Int32(nmaps),
            Int32(rows),
            Int32(cols),
            Int32(npix)
        )

        maxv = CUDA.maximum(out_d)
        if maxv > 0f0
            logmax = log1p(maxv)
            @cuda threads=threads blocks=blocks _normalize_image_kernel!(out_d, Int32(npix), logmax)
        end

        return Fractals.Gray.(Array(out_d))
    end

    colors_h = Fractals._map_colors(nmaps)
    map_cr_h = Vector{Float32}(undef, nmaps)
    map_cg_h = Vector{Float32}(undef, nmaps)
    map_cb_h = Vector{Float32}(undef, nmaps)
    @inbounds for i in 1:nmaps
        c = Fractals._map_color_rgb(i, colors_h)
        map_cr_h[i] = c.r
        map_cg_h[i] = c.g
        map_cb_h[i] = c.b
    end
    map_cr = CuArray(map_cr_h)
    map_cg = CuArray(map_cg_h)
    map_cb = CuArray(map_cb_h)

    r_d = CUDA.zeros(Float32, rows, cols)
    g_d = CUDA.zeros(Float32, rows, cols)
    b_d = CUDA.zeros(Float32, rows, cols)

    @cuda threads=threads blocks=blocks _iterate_image_rgb_kernel!(
        r_d,
        g_d,
        b_d,
        src_d,
        map_a11,
        map_a12,
        map_a21,
        map_a22,
        map_b1,
        map_b2,
        map_cr,
        map_cg,
        map_cb,
        Int32(nmaps),
        Int32(rows),
        Int32(cols),
        Int32(npix)
    )

    maxv = max(CUDA.maximum(r_d), max(CUDA.maximum(g_d), CUDA.maximum(b_d)))
    if maxv > 0f0
        logmax = log1p(maxv)
        @cuda threads=threads blocks=blocks _normalize_rgb_image_kernel!(
            r_d,
            g_d,
            b_d,
            Int32(npix),
            logmax
        )
    end

    return Fractals._rgb_image_from_buffers(Array(r_d), Array(g_d), Array(b_d))
end

end
