# --------------------------------
# Rasterization
# --------------------------------

function make_pixelate_map(limits;
                           resolution=RESOLUTION)

    rows, cols = resolution
    (xlim, ylim) = limits
    xmin, xmax = xlim
    ymin, ymax = ylim

    r = min(rows, cols)
    sx = r / (xmax - xmin)
    sy = r / (ymax - ymin)

    A = SMatrix{2,2,Float64,4}((sx,0.0,
                                0.0,sy))

    b = SVector(
        -xmin*sx + (cols-r)/2 + 0.5,
        -ymin*sy + (rows-r)/2 + 0.5
    )

    return AffineMap(A, b)
end

function _make_image_cpu(ifs::IFS; resolution::Tuple{Int,Int}=RESOLUTION)
    map = make_pixelate_map(ifs.limits; resolution=resolution)
    rows, cols = resolution
    npts = length(ifs.points)
    nthreads_local = _thread_buffer_slots()
    nstripes = _make_image_cpu_stripes(npts, rows, cols)
    buffers = [[zeros(Float32, rows, cols) for _ in 1:nstripes] for _ in 1:nthreads_local]

    @threads for idx in eachindex(ifs.points)
        tid = threadid()
        pt = ifs.points[idx]
        pixels = map(pt)
        pixelx = clamp(round(Int, pixels[1]), 1, cols)
        pixely = clamp(round(Int, pixels[2]), 1, rows)
        stripe = ((idx - 1) % nstripes) + 1

        @inbounds buffers[tid][stripe][pixely, pixelx] += 1.0f0
    end

    img = buffers[1][1]
    for s in 2:nstripes
        img .+= buffers[1][s]
    end
    for t in 2:nthreads_local
        thread_img = buffers[t][1]
        for s in 2:nstripes
            thread_img .+= buffers[t][s]
        end
        img .+= thread_img
    end
    maxv = maximum(img)
    if maxv > 0f0
        logmax = log1p(maxv)
        @inbounds for i in eachindex(img)
            img[i] = log1p(img[i]) / logmax
        end
    end
    return img
end

@inline function _make_image_cpu_stripes(npts::Int, rows::Int, cols::Int)::Int
    density = npts / max(rows * cols, 1)
    if npts >= 500_000 || density >= 8
        return 4
    elseif npts >= 100_000 || density >= 2
        return 2
    end
    return 1
end

function _make_image_gpu(ifs::IFS, ::Val; resolution::Tuple{Int,Int}=RESOLUTION)
    throw(ArgumentError("GPU backend is not available. Install CUDA.jl and ensure a functional CUDA runtime, or use backend=:cpu/:auto."))
end

function make_image(ifs::IFS; resolution::Tuple{Int,Int}=RESOLUTION, backend::Symbol=:cpu)
    if backend == :cpu
        return _make_image_cpu(ifs; resolution=resolution)
    elseif backend == :gpu
        return _make_image_gpu(ifs, Val(:cuda); resolution=resolution)
    elseif backend == :auto
        if _gpu_backend_available(Val(:cuda))
            return _make_image_gpu(ifs, Val(:cuda); resolution=resolution)
        end
        return _make_image_cpu(ifs; resolution=resolution)
    end
    throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))
end
