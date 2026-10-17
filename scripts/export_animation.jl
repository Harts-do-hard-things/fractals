#!/usr/bin/env julia

using DrWatson
@quickactivate "fractals"

using Fractals

function _usage()
    println(
        """
Usage:
  export_animation.jl --frames-dir <dir> --basename <name> [--format gif|mp4] [--out <path>] [--fps <n>] [--ffmpeg <cmd>]
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

function main(args=ARGS)
    opts = _parse_args(args)
    isnothing(opts) && return 0
    frames_dir = get(opts, "frames-dir", nothing)
    basename = get(opts, "basename", nothing)
    outpath = get(opts, "out", nothing)
    isnothing(frames_dir) && throw(ArgumentError("--frames-dir is required"))
    isnothing(basename) && throw(ArgumentError("--basename is required"))
    format = Symbol(lowercase(get(opts, "format", "gif")))
    format in (:gif, :mp4) ||
        throw(ArgumentError("--format must be 'gif' or 'mp4', got '$format'"))
    fps = parse(Float64, get(opts, "fps", "12"))
    ffmpeg_cmd = get(opts, "ffmpeg", "ffmpeg")
    result = export_animation(format;
                              frames_dir=frames_dir,
                              basename=basename,
                              outpath=outpath,
                              fps=fps,
                              ffmpeg_cmd=ffmpeg_cmd)
    println("Wrote $(uppercase(String(format))) to $(result.outpath)")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
