# -------------------------------------------------
# Check if triangle contains origin
# -------------------------------------------------

function isinspace(v::AbstractVector, limits::Tuple{Tuple,Tuple})
    ( xlim, ylim ) = limits
    xmin, xmax = xlim
    ymin, ymax = ylim
    return xmin ≤ v[1] ≤ xmax && ymin ≤ v[2] ≤ ymax
end

function isinspace(point::SVector{2, Float64}, point1::SVector{2, Float64}, point2::SVector{2, Float64}, limits::Tuple{Tuple,Tuple})
    return isinspace(point, limits) || isinspace(point1, limits) || isinspace(point2, limits)
end

@inline function points_contain_zero(
    points::NTuple{3,SVector{2,Float64}}
)::Bool
    p0, p1, p2 = points

    v1 = p1 - p0
    v2 = p2 - p0
    vO = -p0

    return (0 ≤ dot(v1, vO) ≤ dot(v1, v1)) &&
           (0 ≤ dot(v2, vO) ≤ dot(v2, v2))
end


# -------------------------------------------------
# Inverse iteration (no comprehension allocations)
# -------------------------------------------------

function inverse_iterate(
    inverse_maps::Vector{AffineMap{Float64}},
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}},
    n::Integer,
    p0::SVector{2,Float64},
    p1::SVector{2,Float64},
    p2::SVector{2,Float64}
)::Float32

    current = Vector{NTuple{3,SVector{2,Float64}}}()
    push!(current, (p0, p1, p2))

    next = Vector{NTuple{3,SVector{2,Float64}}}()

    for j in 1:n

        empty!(next)

        for tri in current
            if !isinspace(tri[1], limits) &&
               !isinspace(tri[2], limits) &&
               !isinspace(tri[3], limits)
                continue
            end

            for imap in inverse_maps
                newtri = (imap(tri[1]),
                          imap(tri[2]),
                          imap(tri[3]))

                push!(next, newtri)
            end
        end

        if isempty(next)
            return Float32(j / n * 0.5)
        end

        if any(points_contain_zero, next)
            return 1.0f0
        end

        current, next = next, current
    end

    return 0.0f0
end

@inline function _inverse_hide_scale_value(v::Float32)::Float32
    # 0.0 means did not diverge, 1.0 means contains-zero; both are white in hide mode.
    return (v == 0.0f0 || v == 1.0f0) ? 1.0f0 : 0.0f0
end

function _inverse_iterate!(
    current::Vector{NTuple{3,SVector{2,Float64}}},
    next::Vector{NTuple{3,SVector{2,Float64}}},
    inverse_maps::Vector{AffineMap{Float64}},
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}},
    n::Integer,
    p0::SVector{2,Float64},
    p1::SVector{2,Float64},
    p2::SVector{2,Float64};
    max_frontier::Int=typemax(Int)
)::Float32
    empty!(current)
    empty!(next)
    push!(current, (p0, p1, p2))

    for j in 1:n
        empty!(next)

        for tri in current
            if !isinspace(tri[1], limits) &&
               !isinspace(tri[2], limits) &&
               !isinspace(tri[3], limits)
                continue
            end

            for imap in inverse_maps
                if length(next) < max_frontier
                    push!(next, (imap(tri[1]), imap(tri[2]), imap(tri[3])))
                end
            end
        end

        if isempty(next)
            return Float32(j / n * 0.5)
        end

        if any(points_contain_zero, next)
            return 1.0f0
        end

        current, next = next, current
    end

    return 0.0f0
end

function inverse_iterate(
    ifs::IFS,
    n::Integer,
    p0::SVector{2,Float64},
    p1::SVector{2,Float64},
    p2::SVector{2,Float64}
)::Float32

    inverse_maps = inv.(ifs.maps)
    return inverse_iterate(inverse_maps, ifs.limits, n, p0, p1, p2)
end


# -------------------------------------------------
# Inverse rasterization
# -------------------------------------------------

function _rasterize_image_inversely_cpu(
    ifs::IFS,
    n::Integer,
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}};
    resolution::Tuple{Int,Int}=RESOLUTION,
    show_divergence_scale::Bool=true,
    mode::Symbol=:exact
    )

    pixel_map = make_pixelate_map(limits;
                                  resolution=resolution)
    inverse_maps = inv.(ifs.maps)

    inv_pixel_map = inv(pixel_map)

    rows, cols = resolution
    img = zeros(Float32, rows, cols)
    nthreads_local = _thread_buffer_slots()
    max_frontier = _inverse_cpu_capacity(ifs, n, mode)
    current_buffers = [Vector{NTuple{3,SVector{2,Float64}}}() for _ in 1:nthreads_local]
    next_buffers = [Vector{NTuple{3,SVector{2,Float64}}}() for _ in 1:nthreads_local]

    @threads for x in 1:cols
        tid = threadid()
        current = current_buffers[tid]
        next = next_buffers[tid]
        @inbounds for y in 1:rows
            p0 = inv_pixel_map(SVector{2,Float64}(x,     y))
            p1 = inv_pixel_map(SVector{2,Float64}(x + 1, y))
            p2 = inv_pixel_map(SVector{2,Float64}(x,     y + 1))

            v = _inverse_iterate!(current, next, inverse_maps, ifs.limits, n, p0, p1, p2; max_frontier=max_frontier)
            img[y, x] = show_divergence_scale ? v : _inverse_hide_scale_value(v)
        end
    end

    return img
end

@inline function _inverse_cpu_capacity(ifs::IFS, n::Integer, mode::Symbol)::Int
    nmaps = length(ifs.maps)
    nmaps > 0 || throw(ArgumentError("IFS must have at least one map"))
    n >= 0 || throw(ArgumentError("n must be >= 0, got $n"))
    mode in (:exact, :preview) ||
        throw(ArgumentError("Invalid mode '$mode'. Supported: :exact, :preview"))
    if mode == :exact
        return typemax(Int)
    end

    cap = 1
    for _ in 1:Int(n)
        cap *= nmaps
        cap = min(cap, 256)
    end
    return max(16, cap)
end

@inline function _inverse_gpu_exact_capacity(nmaps::Int, n::Int)::Int
    if n == 0
        return 1
    end
    cap = 1
    for _ in 1:n
        cap = cap * nmaps
    end
    return max(1, cap)
end

@inline function _inverse_gpu_preview_capacity(
    nmaps::Int,
    n::Int,
    resolution::Tuple{Int,Int}
)::Int
    rows, cols = resolution
    npix = max(rows * cols, 1)
    # Keep preview mode bounded to reduce GPU memory pressure on larger frames.
    preview_budget_bytes = 192 * 1024 * 1024
    max_by_budget = Int(floor(preview_budget_bytes / (2 * 6 * sizeof(Float64) * npix)))
    max_by_budget = clamp(max_by_budget, 16, 256)
    base = min(_inverse_gpu_exact_capacity(nmaps, n), 256)
    return max(16, min(base, max_by_budget))
end

@inline function _inverse_gpu_buffer_bytes(cap::Int, rows::Int, cols::Int)::Int
    npix = rows * cols
    # 2 ping-pong buffers * 6 Float64 lanes per triangle.
    return 2 * 6 * sizeof(Float64) * cap * npix
end

function _inverse_gpu_capacity(
    ifs::IFS,
    n::Integer,
    resolution::Tuple{Int,Int},
    mode::Symbol
)::Int
    nmaps = length(ifs.maps)
    nmaps > 0 || throw(ArgumentError("IFS must have at least one map"))
    n >= 0 || throw(ArgumentError("n must be >= 0, got $n"))
    n_int = Int(n)
    if mode == :exact
        return _inverse_gpu_exact_capacity(nmaps, n_int)
    elseif mode == :preview
        return _inverse_gpu_preview_capacity(nmaps, n_int, resolution)
    end
    throw(ArgumentError("Invalid mode '$mode'. Supported: :exact, :preview"))
end

function _rasterize_image_inversely_gpu(
    ifs::IFS,
    n::Integer,
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}},
    ::Val;
    resolution::Tuple{Int,Int}=RESOLUTION,
    show_divergence_scale::Bool=true,
    mode::Symbol=:exact
)
    throw(ArgumentError("GPU backend is not available. Install CUDA.jl and ensure a functional CUDA runtime, or use backend=:cpu/:auto."))
end

function _rasterize_image_inversely_gpu_or_throw(
    ifs::IFS,
    n::Integer,
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}};
    resolution::Tuple{Int,Int}=RESOLUTION,
    show_divergence_scale::Bool=true,
    mode::Symbol=:exact
)
    cap = _inverse_gpu_capacity(ifs, n, resolution, mode)
    rows, cols = resolution
    bytes = _inverse_gpu_buffer_bytes(cap, rows, cols)
    max_bytes = 768 * 1024 * 1024
    if bytes > max_bytes
        throw(ArgumentError("Inverse GPU '$mode' buffers are too large for resolution=$resolution and n=$n (estimated $(round(bytes / 1024^2; digits=1)) MiB > $(round(max_bytes / 1024^2; digits=1)) MiB). Reduce resolution/iterations or use backend=:cpu."))
    end

    return _rasterize_image_inversely_gpu(ifs, n, limits, Val(:cuda);
                                          resolution=resolution,
                                          show_divergence_scale=show_divergence_scale,
                                          mode=mode)
end

function rasterize_image_inversely(
    ifs::IFS,
    n::Integer,
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}};
    resolution::Tuple{Int,Int}=RESOLUTION,
    show_divergence_scale::Bool=true,
    backend::Symbol=:cpu,
    mode::Symbol=:exact
    )
    backend in (:cpu, :gpu, :auto) ||
        throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))
    mode in (:exact, :preview) ||
        throw(ArgumentError("Invalid mode '$mode'. Supported: :exact, :preview"))

    if backend == :cpu
        return _rasterize_image_inversely_cpu(ifs, n, limits;
                                              resolution=resolution,
                                              show_divergence_scale=show_divergence_scale,
                                              mode=mode)
    elseif backend == :gpu
        return _rasterize_image_inversely_gpu_or_throw(ifs, n, limits;
                                                       resolution=resolution,
                                                       show_divergence_scale=show_divergence_scale,
                                                       mode=mode)
    elseif _gpu_backend_available(Val(:cuda))
        try
            return _rasterize_image_inversely_gpu_or_throw(ifs, n, limits;
                                                           resolution=resolution,
                                                           show_divergence_scale=show_divergence_scale,
                                                           mode=mode)
        catch err
            if err isa ArgumentError
                return _rasterize_image_inversely_cpu(ifs, n, limits;
                                                      resolution=resolution,
                                                      show_divergence_scale=show_divergence_scale,
                                                      mode=mode)
            end
            rethrow(err)
        end
    end

    return _rasterize_image_inversely_cpu(ifs, n, limits;
                                          resolution=resolution,
                                          show_divergence_scale=show_divergence_scale,
                                          mode=mode)
end
