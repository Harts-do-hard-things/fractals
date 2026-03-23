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
