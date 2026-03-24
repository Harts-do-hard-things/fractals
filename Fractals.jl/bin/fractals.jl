#!/usr/bin/env julia

using Fractals
using DelimitedFiles

include("../bench/benchmark_suite.jl")
using .BenchmarkSuite

function _usage()
    println(
        """
Usage:
  fractals.jl <command> [options]

Commands:
  render        Render one fractal from an .ifs file or matrix file
  batch-render  Render all IFS definitions from one .ifs file
  validate-ifs  Parse and validate an .ifs file
  benchmark     Run a small benchmark suite

Common options:
  --input <path>           Input file path
  --method <method>        Chaos|Parallel|PointDeterministic|ImageIterate|Inverse|RenderTransformations
  --npoints <int>          Number of points
  --iterations <int>       Iterations for deterministic/inverse methods
  --color <bool>           true|false, colorize supported render methods
  --show-divergence-scale  true|false for inverse rendering
  --backend <name>         cpu|gpu|auto (render + benchmark GPU-capable backends)
  --image-source <name>    polygon|chaos|point_deterministic|inverse|file (ImageIterate only)
  --image-path <path>      Source image path when --image-source file
  --image-iterations <int> Number of iterate_image passes (ImageIterate only)
  --polygon-limits-mode    geometry|ifs|default (ImageIterate polygon source)
  --initial-polygon <name> default|equilateral_triangle|line_arrow|line
  --show-base <bool>       true|false for RenderTransformations
  --axis <bool>            true|false for RenderTransformations
  --resolution <HxW>       Image resolution (e.g. 1024x1024)
  --ifs-index <int>        Select IFS by index from an .ifs file
  --ifs-name <name>        Select IFS by name from an .ifs file
  --out <path>             Output image path
  --out-dir <path>         Output directory for batch-render
  --profile <name>         Benchmark profile: small|medium|large|all
  --repeats <int>          Benchmark repeat count
  --json <path>            Benchmark JSON output path
  --targets <path>         Benchmark target envelope TOML file
  --include-gpu-bench      Also run make_image/iterate_image GPU benchmarks (benchmark command)
  --strict                 Fail benchmark command when targets status is fail
"""
    )
end

function _parse_args(args::Vector{String})
    opts = Dict{String,Any}()
    pos = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a[3:end]
            if i < length(args) && !startswith(args[i + 1], "--")
                opts[key] = args[i + 1]
                i += 2
            else
                opts[key] = true
                i += 1
            end
        else
            push!(pos, a)
            i += 1
        end
    end
    return pos, opts
end

function _opt_str(opts::Dict{String,Any}, key::String, default::Union{Nothing,String}=nothing)
    haskey(opts, key) || return default
    v = opts[key]
    return isa(v, String) ? v : string(v)
end

function _opt_int(opts::Dict{String,Any}, key::String, default::Union{Nothing,Int}=nothing)
    haskey(opts, key) || return default
    return parse(Int, string(opts[key]))
end

function _opt_bool(opts::Dict{String,Any}, key::String, default::Bool)
    haskey(opts, key) || return default
    v = lowercase(strip(string(opts[key])))
    if v in ("true", "1", "yes", "y", "on")
        return true
    elseif v in ("false", "0", "no", "n", "off")
        return false
    end
    throw(ArgumentError("Invalid boolean for --$key: '$(opts[key])'. Expected true|false"))
end

function _parse_resolution(raw::String)
    cleaned = replace(strip(raw), "," => "x", "X" => "x")
    parts = split(cleaned, "x")
    length(parts) == 2 || throw(ArgumentError("Invalid resolution '$raw'. Expected HxW, e.g. 1024x1024"))
    h = parse(Int, strip(parts[1]))
    w = parse(Int, strip(parts[2]))
    h > 0 && w > 0 || throw(ArgumentError("Resolution must be positive, got ($h, $w)"))
    return (h, w)
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

function _slug(s::AbstractString)
    t = lowercase(strip(s))
    t = replace(t, r"[^a-z0-9]+" => "_")
    t = replace(t, r"^_+|_+$" => "")
    return isempty(t) ? "ifs" : t
end

function _push_kw!(kwargs::Vector{Pair{Symbol,Any}}, key::Symbol, value)
    isnothing(value) && return kwargs
    push!(kwargs, key => value)
    return kwargs
end

function _render_kwargs(opts::Dict{String,Any}; outpath::Union{Nothing,String}=nothing, ifs_index::Union{Nothing,Int}=nothing, ifs_name::Union{Nothing,String}=nothing)
    method = _opt_str(opts, "method", "Chaos")
    parsed_method = Fractals._parse_render_method(method)
    render_method = parsed_method == Parallel ? Chaos : parsed_method
    backend = Symbol(_opt_str(opts, "backend", "cpu"))
    npoints = _opt_int(opts, "npoints", nothing)
    iterations = _opt_int(opts, "iterations", nothing)
    color = _opt_bool(opts, "color", false)
    show_divergence_scale = _opt_bool(opts, "show-divergence-scale", true)
    image_source = Symbol(_opt_str(opts, "image-source", "polygon"))
    image_path = _opt_str(opts, "image-path", nothing)
    image_iterations = _opt_int(opts, "image-iterations", 1)
    polygon_limits_mode = Symbol(_opt_str(opts, "polygon-limits-mode", "geometry"))
    initial_polygon = Symbol(_opt_str(opts, "initial-polygon", "default"))
    show_base = _opt_bool(opts, "show-base", false)
    axis = _opt_bool(opts, "axis", false)
    resolution = haskey(opts, "resolution") ? _parse_resolution(string(opts["resolution"])) : RESOLUTION

    kwargs = Pair{Symbol,Any}[
        :method => method,
        :color => color,
        :resolution => resolution,
    ]
    _push_kw!(kwargs, :npoints, npoints)
    _push_kw!(kwargs, :ifs_index, ifs_index)
    _push_kw!(kwargs, :ifs_name, ifs_name)
    _push_kw!(kwargs, :outpath, outpath)

    if render_method in (Chaos, PointDeterministic, Inverse, ImageIterate)
        push!(kwargs, :backend => backend)
    end
    if render_method in (PointDeterministic, Inverse, ImageIterate)
        _push_kw!(kwargs, :iterations, iterations)
    end
    if render_method == Inverse
        push!(kwargs, :show_divergence_scale => show_divergence_scale)
    elseif render_method == ImageIterate
        push!(kwargs, :show_divergence_scale => show_divergence_scale)
        push!(kwargs, :image_source => image_source)
        push!(kwargs, :image_iterations => image_iterations)
        push!(kwargs, :polygon_limits_mode => polygon_limits_mode)
        push!(kwargs, :initial_polygon => initial_polygon)
        push!(kwargs, :show_base => show_base)
        push!(kwargs, :axis => axis)
        _push_kw!(kwargs, :image_path, image_path)
    elseif render_method == RenderTransformations
        push!(kwargs, :initial_polygon => initial_polygon)
        push!(kwargs, :show_base => show_base)
        push!(kwargs, :axis => axis)
    end

    return kwargs
end

function _render_from_input(input::String, opts::Dict{String,Any})
    image_source = Symbol(_opt_str(opts, "image-source", "polygon"))
    initial_polygon = Symbol(_opt_str(opts, "initial-polygon", "default"))
    ifs_index = _opt_int(opts, "ifs-index", nothing)
    ifs_name = _opt_str(opts, "ifs-name", nothing)
    outpath = _opt_str(opts, "out", "media/cli_render.png")
    if image_source != :polygon && initial_polygon != :default
        @warn "--initial-polygon is ignored unless --image-source polygon."
    end
    kwargs = _render_kwargs(opts; outpath=outpath, ifs_index=ifs_index, ifs_name=ifs_name)

    if endswith(lowercase(input), ".ifs")
        return render(input; kwargs...)
    end

    eq = _read_eq_file(input)
    return render(eq; kwargs...)
end

function _cmd_render(opts::Dict{String,Any})
    input = _opt_str(opts, "input", nothing)
    isnothing(input) && throw(ArgumentError("render requires --input <path>"))
    out = _render_from_input(input, opts)
    println("Rendered $(out.method) to $(out.outpath)")
    return 0
end

function _cmd_batch_render(opts::Dict{String,Any})
    input = _opt_str(opts, "input", nothing)
    isnothing(input) && throw(ArgumentError("batch-render requires --input <path-to-.ifs>"))
    endswith(lowercase(input), ".ifs") || throw(ArgumentError("batch-render currently supports only .ifs input"))

    image_source = Symbol(_opt_str(opts, "image-source", "polygon"))
    initial_polygon = Symbol(_opt_str(opts, "initial-polygon", "default"))
    outdir = _opt_str(opts, "out-dir", "media/batch")
    mkpath(outdir)

    defs = parse_ifs_file(input; npoints=1)
    isempty(defs) && throw(ArgumentError("No IFS definitions found in '$input'"))

    println("Batch rendering $(length(defs)) definitions from $input")
    if image_source != :polygon && initial_polygon != :default
        @warn "--initial-polygon is ignored unless --image-source polygon."
    end
    for (i, d) in enumerate(defs)
        filename = lpad(string(i), 2, '0') * "_" * _slug(d.name) * ".png"
        outpath = joinpath(outdir, filename)
        kwargs = _render_kwargs(opts; ifs_index=i, outpath=outpath)
        out = render(input; kwargs...)
        println("[$i] $(d.name) -> $(out.outpath)")
    end
    return 0
end

function _cmd_validate_ifs(opts::Dict{String,Any})
    input = _opt_str(opts, "input", nothing)
    isnothing(input) && throw(ArgumentError("validate-ifs requires --input <path-to-.ifs>"))
    defs = Fractals.parse_ifs_definitions_file(input)
    isempty(defs) && throw(ArgumentError("No IFS definitions found in '$input'"))
    valid, invalid = Fractals._split_valid_ifs_definitions(defs)

    if isempty(invalid)
        println("Valid IFS file: $input")
        println("Definitions: $(length(valid))")
        for (i, d) in enumerate(valid)
            println("  [$i] $(d.name)")
        end
        return 0
    end

    println(stderr, "Invalid IFS file: $input")
    println(stderr, "Valid definitions: $(length(valid))")
    println(stderr, "Invalid definitions: $(length(invalid))")
    for entry in invalid
        println(stderr, "  $(entry.definition.name)")
        for failure in entry.failures
            println(stderr, "    - $(Fractals._format_contractivity_failure(failure))")
        end
    end
    throw(ArgumentError("IFS file contains non-contractive definitions"))
end

function _cmd_benchmark(opts::Dict{String,Any})
    profile = _opt_str(opts, "profile", "small")
    backend = Symbol(_opt_str(opts, "backend", "cpu"))
    include_gpu_bench = get(opts, "include-gpu-bench", false) == true
    repeats = _opt_int(opts, "repeats", 3)
    json_path = _opt_str(opts, "json", nothing)
    targets_path = _opt_str(opts, "targets", nothing)
    strict = get(opts, "strict", false) == true
    payload = BenchmarkSuite.run_suite(; profile=profile,
                                         repeats=repeats,
                                         backend=backend,
                                         include_gpu_bench=include_gpu_bench,
                                         json_path=json_path,
                                         targets_path=targets_path,
                                         strict=strict)
    BenchmarkSuite.print_report(payload)
    if !isnothing(json_path)
        println("Wrote benchmark JSON to $json_path")
    end
    return 0
end

function main(args::Vector{String}=ARGS)
    pos, opts = _parse_args(args)
    isempty(pos) && begin
        _usage()
        return 1
    end

    cmd = pos[1]
    if cmd == "render"
        return _cmd_render(opts)
    elseif cmd == "batch-render"
        return _cmd_batch_render(opts)
    elseif cmd == "validate-ifs"
        return _cmd_validate_ifs(opts)
    elseif cmd == "benchmark"
        return _cmd_benchmark(opts)
    elseif cmd in ("-h", "--help", "help")
        _usage()
        return 0
    end

    throw(ArgumentError("Unknown command '$cmd'"))
end

try
    exit(main())
catch e
    println(stderr, "ERROR: ", sprint(showerror, e))
    exit(2)
end
