module FractalsCUDAExt

using CUDA
using Fractals

import Fractals: IFS, RESOLUTION, _gpu_backend_available, _make_image_gpu

@inline function _gpu_backend_available(::Val{:cuda})
    try
        return CUDA.functional()
    catch
        return false
    end
end

function _rasterize_points_kernel!(
    img,
    xs,
    ys,
    n::Int32,
    rows::Int32,
    cols::Int32,
    sx::Float32,
    sy::Float32,
    bx::Float32,
    by::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
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
        CUDA.@atomic img[py, px] += 1.0f0
    end
    return
end

function _normalize_image_kernel!(img, n::Int32, logmax::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds img[i] = log1p(img[i]) / logmax
    end
    return
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
    img_d = CUDA.zeros(Float32, rows, cols)

    threads = 256
    blocks_points = cld(npts, threads)
    @cuda threads=threads blocks=blocks_points _rasterize_points_kernel!(
        img_d,
        xs,
        ys,
        Int32(npts),
        Int32(rows),
        Int32(cols),
        sx,
        sy,
        bx,
        by
    )

    maxv = CUDA.maximum(img_d)
    if maxv > 0f0
        logmax = log1p(maxv)
        npix = length(img_d)
        blocks_pixels = cld(npix, threads)
        @cuda threads=threads blocks=blocks_pixels _normalize_image_kernel!(img_d, Int32(npix), logmax)
    end

    return Array(img_d)
end

end
