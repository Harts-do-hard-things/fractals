function _render_method_symbol(m::RenderMethod)
    if m == PointDeterministic
        return :point_deterministic
    elseif m == ImageIterate
        return :image_iterate
    elseif m == RenderTransformations
        return :render_transformations
    end
    return Symbol(lowercase(string(m)))
end

function _parse_render_method(method::RenderMethod)
    return method
end

function _parse_render_method(method::Symbol)
    m = Symbol(lowercase(String(method)))
    if m == :chaos
        return Chaos
    elseif m == :parallel
        return Parallel
    elseif m == :point_deterministic || m == :pointdeterministic
        return PointDeterministic
    elseif m == :image_iterate || m == :imageiterate
        return ImageIterate
    elseif m == :inverse
        return Inverse
    elseif m == :render_transformations || m == :rendertransformations || m == :transformations
        return RenderTransformations
    elseif m == :deterministic
        throw(ArgumentError("Method '$method' was removed. Use point_deterministic."))
    end
    throw(ArgumentError("Invalid method '$method'. Supported methods: chaos, parallel, point_deterministic, image_iterate, inverse, render_transformations"))
end

function _parse_render_method(method::AbstractString)
    return _parse_render_method(Symbol(lowercase(strip(method))))
end

function _normalize_media_outpath(outpath::AbstractString)
    raw = String(outpath)
    media_prefix = string(DEFAULT_MEDIA_DIR, Base.Filesystem.path_separator)
    normalized = normpath(raw)

    if normalized == DEFAULT_MEDIA_DIR || startswith(normalized, media_prefix)
        mkpath(dirname(normalized))
        return normalized
    end

    final = joinpath(DEFAULT_MEDIA_DIR, basename(normalized))
    mkpath(dirname(final))
    return final
end

const _PNG_SIGNATURE = UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
const _PNG_SOURCE_LIMITS_KEYWORD = "fractals.source_limits"
const _PNG_SOURCE_LIMITS_VERSION = 1
const _PNG_CRC32_TABLE = let table = Vector{UInt32}(undef, 256)
    for i in 0:255
        crc = UInt32(i)
        for _ in 1:8
            if isodd(crc)
                crc = (crc >> 1) ⊻ 0xedb88320
            else
                crc >>= 1
            end
        end
        table[i + 1] = crc
    end
    table
end

@inline function _has_png_extension(path::AbstractString)
    return lowercase(splitext(String(path))[2]) == ".png"
end

@inline function _limits_tuple(
    x1::Real,
    x2::Real,
    y1::Real,
    y2::Real,
)
    return ((Float64(x1), Float64(x2)), (Float64(y1), Float64(y2)))
end

function _serialize_source_limits_payload(limits)
    return JSON3.write((
        version=_PNG_SOURCE_LIMITS_VERSION,
        x=[limits[1][1], limits[1][2]],
        y=[limits[2][1], limits[2][2]],
    ))
end

function _deserialize_source_limits_payload(payload::AbstractString)
    obj = JSON3.read(payload)
    version = get(obj, :version, get(obj, "version", nothing))
    version == _PNG_SOURCE_LIMITS_VERSION ||
        throw(ArgumentError("Unsupported source limits metadata version '$version'"))

    x = get(obj, :x, get(obj, "x", nothing))
    y = get(obj, :y, get(obj, "y", nothing))
    (x === nothing || y === nothing || length(x) != 2 || length(y) != 2) &&
        throw(ArgumentError("Invalid source limits metadata payload"))

    return _limits_tuple(x[1], x[2], y[1], y[2])
end

@inline function _png_read_u32be(bytes::AbstractVector{UInt8}, pos::Int)
    return (UInt32(bytes[pos]) << 24) |
           (UInt32(bytes[pos + 1]) << 16) |
           (UInt32(bytes[pos + 2]) << 8) |
           UInt32(bytes[pos + 3])
end

function _png_write_u32be(value::UInt32)
    return UInt8[
        UInt8((value >> 24) & 0xff),
        UInt8((value >> 16) & 0xff),
        UInt8((value >> 8) & 0xff),
        UInt8(value & 0xff),
    ]
end

function _png_crc32(bytes::AbstractVector{UInt8})
    crc = 0xffffffff % UInt32
    for b in bytes
        idx = Int(((crc ⊻ UInt32(b)) & 0xff) + 0x01)
        crc = (crc >> 8) ⊻ _PNG_CRC32_TABLE[idx]
    end
    return ~crc
end

@inline function _is_png_bytes(bytes::AbstractVector{UInt8})
    length(bytes) >= length(_PNG_SIGNATURE) || return false
    return bytes[1:length(_PNG_SIGNATURE)] == _PNG_SIGNATURE
end

function _parse_png_chunks(bytes::AbstractVector{UInt8})
    _is_png_bytes(bytes) || throw(ArgumentError("File is not a valid PNG"))
    chunks = NamedTuple[]
    pos = length(_PNG_SIGNATURE) + 1

    while pos + 11 <= length(bytes)
        raw_start = pos
        len = Int(_png_read_u32be(bytes, pos))
        type_start = pos + 4
        data_start = pos + 8
        data_end = data_start + len - 1
        crc_end = data_end + 4
        crc_end <= length(bytes) || throw(ArgumentError("Truncated PNG chunk"))

        chunk_type = String(Char.(bytes[type_start:type_start + 3]))
        chunk_data = len == 0 ? UInt8[] : Vector{UInt8}(bytes[data_start:data_end])
        raw = Vector{UInt8}(bytes[raw_start:crc_end])
        push!(chunks, (type=chunk_type, data=chunk_data, raw=raw))

        pos = crc_end + 1
        chunk_type == "IEND" && break
    end

    isempty(chunks) && throw(ArgumentError("PNG contains no chunks"))
    return chunks
end

function _png_text_keyword_and_value(data::AbstractVector{UInt8})
    nul = findfirst(==(0x00), data)
    nul === nothing && return nothing
    keyword = String(Char.(data[1:nul-1]))
    value = String(Char.(data[nul + 1:end]))
    return keyword => value
end

function _png_make_text_chunk(keyword::AbstractString, value::AbstractString)
    keyword_bytes = Vector{UInt8}(codeunits(String(keyword)))
    value_bytes = Vector{UInt8}(codeunits(String(value)))
    data = UInt8[keyword_bytes; 0x00; value_bytes]
    type_bytes = UInt8[0x74, 0x45, 0x58, 0x74] # tEXt
    crc = _png_crc32(UInt8[type_bytes; data])
    return UInt8[_png_write_u32be(UInt32(length(data))); type_bytes; data; _png_write_u32be(crc)]
end

function _embed_png_text_chunk(path::AbstractString, keyword::AbstractString, value::AbstractString)
    bytes = read(path)
    chunks = _parse_png_chunks(bytes)
    out = copy(_PNG_SIGNATURE)
    inserted = false

    for chunk in chunks
        keep = true
        if chunk.type == "tEXt"
            pair = _png_text_keyword_and_value(chunk.data)
            if !isnothing(pair) && first(pair) == keyword
                keep = false
            end
        end

        if keep
            append!(out, chunk.raw)
        end

        if !inserted && chunk.type == "IHDR"
            append!(out, _png_make_text_chunk(keyword, value))
            inserted = true
        end
    end

    inserted || throw(ArgumentError("PNG is missing IHDR chunk"))
    write(path, out)
    return path
end

function _read_png_text_chunk(path::AbstractString, keyword::AbstractString)
    _has_png_extension(path) || return nothing
    isfile(path) || return nothing

    try
        chunks = _parse_png_chunks(read(path))
        for chunk in chunks
            chunk.type == "tEXt" || continue
            pair = _png_text_keyword_and_value(chunk.data)
            if !isnothing(pair) && first(pair) == keyword
                return last(pair)
            end
        end
    catch err
        @warn "Failed to read PNG metadata from '$path'; falling back to default limits." exception=(err, catch_backtrace())
    end

    return nothing
end

function _read_png_source_limits(path::AbstractString)
    payload = _read_png_text_chunk(path, _PNG_SOURCE_LIMITS_KEYWORD)
    isnothing(payload) && return nothing

    try
        return _deserialize_source_limits_payload(payload)
    catch err
        @warn "Failed to parse source limits metadata from '$path'; falling back to default limits." exception=(err, catch_backtrace())
        return nothing
    end
end

function _save_image_with_source_limits(
    outpath::AbstractString,
    img,
    limits,
)
    save(outpath, img)
    if _has_png_extension(outpath)
        _embed_png_text_chunk(outpath, _PNG_SOURCE_LIMITS_KEYWORD, _serialize_source_limits_payload(limits))
    end
    return outpath
end

function _ifs_with_limits(ifs, limits)
    return IFS(ifs.name, ifs.docs, copy(ifs.points), ifs.maps, ifs.weights, limits)
end

@inline function _initial_polygon_names_text()
    return join(string.(supported_initial_polygons()), ", ")
end

function supported_initial_polygons()
    return [preset.name for preset in _INITIAL_POLYGON_REGISTRY]
end

function initial_polygon(name::Symbol=:default)
    normalized = Symbol(lowercase(String(name)))
    for preset in _INITIAL_POLYGON_REGISTRY
        if preset.name == normalized
            return preset
        end
    end
    throw(ArgumentError("Invalid initial_polygon '$name'. Supported: $(_initial_polygon_names_text())"))
end

@inline function _resolve_initial_polygon(preset::InitialPolygonPreset)
    return preset
end

@inline function _resolve_initial_polygon(name::Symbol)
    return initial_polygon(name)
end

@inline function _base_limits_image(initial_polygon_spec::Union{InitialPolygonPreset,Symbol}=:default)
    return _resolve_initial_polygon(initial_polygon_spec).limits
end

@inline function _base_l_image(initial_polygon_spec::Union{InitialPolygonPreset,Symbol}=:default)
    return _resolve_initial_polygon(initial_polygon_spec).segments
end

@inline function _thread_buffer_slots()
    return max(nthreads(), Base.Threads.maxthreadid())
end

@inline function _gpu_backend_available(::Val{:cuda})
    return false
end

@inline function _gpu_backend_unavailable_error()
    return ArgumentError("GPU backend is not available. Install CUDA.jl and ensure a functional CUDA runtime, or use backend=:cpu/:auto.")
end

@inline function _validate_backend(backend::Symbol)
    backend in (:cpu, :gpu, :auto) ||
        throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))
    return backend
end

function _dispatch_backend(
    backend::Symbol,
    cpu_fn,
    gpu_fn;
    gpu_available::Bool=_gpu_backend_available(Val(:cuda)),
    auto_fallback_exceptions::Tuple{Vararg{DataType}}=(ArgumentError,),
)
    _validate_backend(backend)

    if backend == :cpu
        return cpu_fn()
    elseif backend == :gpu
        gpu_available || throw(_gpu_backend_unavailable_error())
        return gpu_fn()
    end

    if !gpu_available
        return cpu_fn()
    end

    try
        return gpu_fn()
    catch err
        if any(T -> err isa T, auto_fallback_exceptions)
            return cpu_fn()
        end
        rethrow(err)
    end
end

function _map_colors(n::Integer)
    n <= 0 && return RGB{Float32}[]
    # Generate visually distinct, deterministic colors and exclude near-white/near-black colors.
    candidates = distinguishable_colors(max(3n, n + 8), [RGB(1, 1, 1), RGB(0, 0, 0)])
    palette = RGB{Float32}[]
    sizehint!(palette, n)
    for c in candidates
        # Relative luminance in sRGB space; keep mid-range tones only.
        lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
        if 0.15 < lum < 0.85
            push!(palette, RGB{Float32}(Float32(c.r), Float32(c.g), Float32(c.b)))
            length(palette) == n && break
        end
    end
    # Fallback to initial candidates if filtering was too aggressive for small n.
    if length(palette) < n
        for c in candidates
            push!(palette, RGB{Float32}(Float32(c.r), Float32(c.g), Float32(c.b)))
            length(palette) == n && break
        end
    end
    return palette
end

@inline function _map_color_rgb(i::Integer, colors::Vector{RGB{Float32}})
    return colors[i]
end

@inline function _rgb_to_hex(c::RGB{Float32})
    r = round(Int, clamp(c.r, 0.0f0, 1.0f0) * 255)
    g = round(Int, clamp(c.g, 0.0f0, 1.0f0) * 255)
    b = round(Int, clamp(c.b, 0.0f0, 1.0f0) * 255)
    return @sprintf("#%02X%02X%02X", r, g, b)
end

function _apply_alpha_mask(img::AbstractMatrix{Float32})
    out = Matrix{GrayA{Float32}}(undef, size(img)...)
    @inbounds for i in eachindex(img)
        value = img[i]
        out[i] = GrayA{Float32}(value, value > 0f0 ? 1f0 : 0f0)
    end
    return out
end

function _apply_alpha_mask(img::AbstractMatrix{Gray{Float32}})
    out = Matrix{GrayA{Float32}}(undef, size(img)...)
    @inbounds for i in eachindex(img)
        value = Float32(img[i])
        out[i] = GrayA{Float32}(value, value > 0f0 ? 1f0 : 0f0)
    end
    return out
end

function _apply_alpha_mask(img::AbstractMatrix{RGB{Float32}})
    out = Matrix{RGBA{Float32}}(undef, size(img)...)
    @inbounds for i in eachindex(img)
        px = img[i]
        a = (px.r > 0f0 || px.g > 0f0 || px.b > 0f0) ? 1f0 : 0f0
        out[i] = RGBA{Float32}(px.r, px.g, px.b, a)
    end
    return out
end
