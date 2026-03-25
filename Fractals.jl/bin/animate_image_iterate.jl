#!/usr/bin/env julia

module AnimateImageIterateScript

using Fractals
using Colors
using FileIO
using Printf
using StaticArrays
using StatsBase: Weights

function _usage()
    println(
        """
Usage:
  animate_image_iterate.jl --input <ifs-path> --image-path <graya.png> [options]

Options:
  --input <path>           Input .ifs or matrix file
  --image-path <path>      GrayA source image path
  --ifs-index <int>        Select IFS by index from an .ifs file
  --ifs-name <name>        Select IFS by name from an .ifs file
  --frames-per-map <int>   Interpolation frames per map stage (default 8)
  --interpolation-mode <m> Map interpolation mode: rotation_scale|linear (default rotation_scale)
  --fps <n>                GIF frame rate (default 12)
  --basename <name>        Frame basename (default image_iterate_build)
  --out-dir <path>         Output frames directory (default media/frames)
  --out <path>             Output GIF path (default media/image_iterate_build.gif)
"""
    )
end

function _parse_args(args::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help")
            _usage()
            return nothing
        end
        startswith(a, "--") || throw(ArgumentError("Unexpected positional argument '$a'"))
        i == length(args) && throw(ArgumentError("Missing value for option '$a'"))
        opts[a[3:end]] = args[i + 1]
        i += 2
    end
    return opts
end

function _read_eq_file(path::String)
    flat_vals = Float64[]
    width = 0
    nrows = 0
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        toks = split(replace(s, "," => " "))
        vals = Float64[parse(Float64, t) for t in toks]
        row_width = length(vals)
        if width == 0
            width = row_width
        elseif row_width != width
            throw(ArgumentError("Inconsistent matrix row width in '$path': expected $width values, got $row_width"))
        end
        append!(flat_vals, vals)
        nrows += 1
    end
    nrows == 0 && throw(ArgumentError("Matrix file '$path' is empty"))
    eq = Matrix{Float64}(undef, nrows, width)
    k = 1
    @inbounds for r in 1:nrows
        for c in 1:width
            eq[r, c] = flat_vals[k]
            k += 1
        end
    end
    return eq
end

function _load_ifs(
    input::AbstractString;
    ifs_index::Union{Nothing,Integer}=nothing,
    ifs_name::Union{Nothing,AbstractString}=nothing,
)
    if endswith(lowercase(String(input)), ".ifs")
        defs = Fractals.parse_ifs_definitions_file(String(input))
        isempty(defs) && throw(ArgumentError("No IFS definitions found in '$input'"))
        if !isnothing(ifs_name)
            idx = findfirst(d -> String(d.name) == String(ifs_name), defs)
            idx === nothing && throw(ArgumentError("IFS name '$(ifs_name)' was not found in '$input'"))
            selected = defs[idx]
        else
            idx = isnothing(ifs_index) ? 1 : Int(ifs_index)
            1 <= idx <= length(defs) || throw(ArgumentError("ifs_index $idx is out of range for '$input'"))
            selected = defs[idx]
        end
        return IFS(selected.eq; npoints=1, name=selected.name, docs=selected.docs)
    end

    !isnothing(ifs_index) && throw(ArgumentError("--ifs-index is only valid for .ifs inputs"))
    !isnothing(ifs_name) && throw(ArgumentError("--ifs-name is only valid for .ifs inputs"))
    return IFS(_read_eq_file(String(input)); npoints=1)
end

function _split_graya_seed(img::AbstractMatrix)
    eltype(img) <: TransparentColor ||
        throw(ArgumentError("animate_image_iterate requires a GrayA source image"))
    background = Matrix{Float32}(undef, size(img)...)
    foreground = Matrix{Float32}(undef, size(img)...)

    @inbounds for i in eachindex(img)
        px = img[i]
        background[i] = Float32(Gray(px))
        foreground[i] = Float32(alpha(px))
    end

    return background, foreground
end

function _gray_to_rgb(img::AbstractMatrix{Float32})
    out = Matrix{RGB{Float32}}(undef, size(img)...)
    @inbounds for i in eachindex(img)
        value = img[i]
        out[i] = RGB{Float32}(value, value, value)
    end
    return out
end

function _single_map_ifs(
    ifs::IFS,
    map_index::Integer,
    t::Real;
    interpolation_mode::Union{Symbol,AbstractString}=:rotation_scale,
)
    map = ifs.maps[map_index]
    left_eq = reshape([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 1, 7)
    right_eq = reshape([map.A[1, 1], map.A[1, 2], map.A[2, 1], map.A[2, 2], map.b[1], map.b[2], 1.0], 1, 7)
    eq = Fractals.interpolate_eq_matrix(left_eq, right_eq, t; interpolation_mode=interpolation_mode)
    maps, weights = Fractals._build_maps_and_weights(eq)
    return IFS(maps, weights; npoints=1, name=ifs.name, docs=ifs.docs, limits=ifs.limits)
end

function _single_map_layer(
    ifs::IFS,
    foreground::AbstractMatrix{Float32},
    map_index::Integer,
    t::Real,
    interpolation_mode::Union{Symbol,AbstractString}=:rotation_scale,
)
    layer_ifs = _single_map_ifs(ifs, map_index, t; interpolation_mode=interpolation_mode)
    return Fractals._iterate_image_single_map(layer_ifs, foreground, 1)
end

function _composite_tinted_layer!(
    dst::AbstractMatrix{RGB{Float32}},
    layer::AbstractMatrix{Gray{Float32}},
    color::RGB{Float32},
)
    @inbounds for i in eachindex(dst)
        α = Float32(layer[i])
        α <= 0f0 && continue
        base = dst[i]
        dst[i] = RGB{Float32}(
            (1 - α) * base.r + α * color.r,
            (1 - α) * base.g + α * color.g,
            (1 - α) * base.b + α * color.b,
        )
    end
    return dst
end

function render_image_iterate_build_frames(
    ifs::IFS,
    image_path::AbstractString;
    frames_per_map::Integer=8,
    outdir::AbstractString=joinpath("media", "frames"),
    basename::AbstractString="image_iterate_build",
    interpolation_mode::Union{Symbol,AbstractString}=:rotation_scale,
)
    frames_per_map > 0 || throw(ArgumentError("frames_per_map must be > 0, got $frames_per_map"))
    !isempty(strip(String(basename))) || throw(ArgumentError("basename must be non-empty"))

    img = load(String(image_path))
    background, foreground = _split_graya_seed(img)
    base_rgb = _gray_to_rgb(background)
    map_colors = Fractals._map_colors(length(ifs.maps))

    mkpath(String(outdir))
    paths = String[]
    stage_ranges = UnitRange{Int}[]
    completed = copy(base_rgb)
    frame_index = 1

    for map_index in 1:length(ifs.maps)
        stage_start = frame_index
        ts = frames_per_map == 1 ? (1.0,) : Tuple(LinRange(0.0, 1.0, frames_per_map))
        for t in ts
            frame = copy(completed)
            layer = _single_map_layer(ifs, foreground, map_index, t, interpolation_mode)
            _composite_tinted_layer!(frame, layer, Fractals._map_color_rgb(map_index, map_colors))
            path = joinpath(String(outdir), @sprintf("%s_%04d.png", basename, frame_index))
            Fractals._save_image_with_source_limits(path, frame, ifs.limits)
            push!(paths, path)
            frame_index += 1
        end

        final_layer = _single_map_layer(ifs, foreground, map_index, 1.0, interpolation_mode)
        _composite_tinted_layer!(completed, final_layer, Fractals._map_color_rgb(map_index, map_colors))
        push!(stage_ranges, stage_start:(frame_index - 1))
    end

    return (
        outdir=String(outdir),
        basename=strip(String(basename)),
        paths=paths,
        stage_ranges=stage_ranges,
        final_image=completed,
    )
end

function create_animation(args=ARGS)
    opts = _parse_args(args)
    isnothing(opts) && return nothing

    input = get(opts, "input", nothing)
    image_path = get(opts, "image-path", nothing)
    isnothing(input) && throw(ArgumentError("--input is required"))
    isnothing(image_path) && throw(ArgumentError("--image-path is required"))

    ifs_index = haskey(opts, "ifs-index") ? parse(Int, opts["ifs-index"]) : nothing
    ifs_name = get(opts, "ifs-name", nothing)
    frames_per_map = parse(Int, get(opts, "frames-per-map", "8"))
    interpolation_mode = get(opts, "interpolation-mode", "rotation_scale")
    fps = parse(Float64, get(opts, "fps", "12"))
    outdir = get(opts, "out-dir", joinpath("media", "frames"))
    basename = get(opts, "basename", "image_iterate_build")
    outpath = get(opts, "out", joinpath("media", basename * ".gif"))

    ifs = _load_ifs(input; ifs_index=ifs_index, ifs_name=ifs_name)
    frames = render_image_iterate_build_frames(ifs, image_path;
                                               frames_per_map=frames_per_map,
                                               outdir=outdir,
                                               basename=basename,
                                               interpolation_mode=interpolation_mode)
    gif = export_animation(:gif;
                           frames_dir=frames.outdir,
                           basename=frames.basename,
                           outpath=outpath,
                           fps=fps)
    return (frames=frames, gif=gif)
end

function main(args=ARGS)
    result = create_animation(args)
    isnothing(result) && return 0
    println("Wrote GIF to $(result.gif.outpath)")
    return 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(AnimateImageIterateScript.main())
end
