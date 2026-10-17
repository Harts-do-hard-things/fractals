# Opt-in benchmarks for the core kernels and every render method. Never run by CI or the tests.
#
#   julia --startup-file=no --project=. scripts/benchmark.jl [small|medium]
#
# Each run saves ONE record to data/sims/benchmarks/ (DrWatson `tagsave`, so it carries the git
# commit), named from the profile and the date/time so runs never overwrite each other. Compare
# two runs from a REPL (`include("scripts/benchmark.jl")`):
#
#   latest_benchmarks()                     # newest records, newest first
#   compare_benchmarks(older_path, newer_path)
#
# All render output goes to a temp root, so benchmark renders never touch plots/ or data/.

using DrWatson
@quickactivate "fractals"

using Fractals, BenchmarkTools, Dates, JLD2, Printf, Statistics

# Sizes per profile: chaos points, render resolution, inverse depth, deterministic points, the
# resolutions for the standalone make_image cases, and the BenchmarkTools budget per case.
const PROFILES = Dict(
    :small => (;
        npoints = 100_000,
        resolution = (256, 256),
        inverse_iterations = 5,
        det_npoints = 200,
        image_resolutions = [(128, 128), (256, 256), (512, 512)],
        seconds = 2.0,
        samples = 20,
    ),
    :medium => (;
        npoints = 1_000_000,
        resolution = (512, 512),
        inverse_iterations = 8,
        det_npoints = 1_000,
        image_resolutions = [(256, 256), (512, 512), (1024, 1024)],
        seconds = 5.0,
        samples = 30,
    ),
)

benchmarks_dir() = datadir("sims", "benchmarks")

# One (name => thunk) pair per case. `tmp` is the scratch root for everything the cases write.
function benchmark_cases(p, tmp::AbstractString)
    ifs = IFS(HEIGHWAY_DRAGON; name = "Heighway Dragon")
    chaos = Chaos(npoints = p.npoints, seed = 1)
    det = PointDeterministic(npoints = p.det_npoints, iterations = 2, seed = 1)
    inverse = Inverse(iterations = p.inverse_iterations)
    points = chaos_game(ifs, p.npoints; seed = 1)

    cases = Pair{String,Any}[]
    add!(name, f) = push!(cases, name => f)

    add!("chaos_game", () -> chaos_game(ifs, p.npoints; seed = 1))

    # A cold miss recomputes and writes the cloud; a warm hit only loads it.
    add!("cached_points/cold", () -> cached_points(chaos, ifs; force = true))
    add!("cached_points/warm", () -> cached_points(chaos, ifs))

    for res in p.image_resolutions
        add!(
            "make_image/$(res[1])x$(res[2])",
            () -> make_image(points, ifs.limits; resolution = res),
        )
    end

    methods = [
        "chaos" => chaos,
        "point_deterministic" => det,
        "inverse" => inverse,
        "transformations" => Transformations(),
        "image_iterate-polygon" => ImageIterate(),
        "image_iterate-chaos" => ImageIterate(chaos; iterations = 2),
    ]
    for (name, m) in methods
        r = Render(ifs, m; resolution = p.resolution)
        add!("draw/$name", () -> Fractals.draw(r))
        add!("render/$name", () -> render(r))
    end

    libdir = joinpath(tmp, "ifs_library")
    add!(
        "build_ifs_library",
        () -> build_ifs_library(; dir = projectdir("ifs"), outdir = libdir, force = true),
    )
    return cases
end

# Plain Dict, so a record loads without BenchmarkTools.
function summarize(trial)
    med, mn = median(trial), minimum(trial)
    return Dict{String,Any}(
        "median_ns" => med.time,
        "min_ns" => mn.time,
        "allocs" => mn.allocs,
        "memory_bytes" => mn.memory,
        "samples" => length(trial),
    )
end

function _format_ns(ns)
    ns < 1e3 && return @sprintf("%.0f ns", ns)
    ns < 1e6 && return @sprintf("%.2f us", ns / 1e3)
    ns < 1e9 && return @sprintf("%.2f ms", ns / 1e6)
    return @sprintf("%.2f s", ns / 1e9)
end

"""
    run_benchmarks(; profile = :small, save = true, only = nothing) -> Dict

Benchmark the chaos game, the point cache, `make_image`, every render method (`draw` and
`render`) and `build_ifs_library` at the `:small` or `:medium` size. With `save = true`, writes
one record to `data/sims/benchmarks/` and stores its path under `"path"`. `only` is an optional
`Regex` that selects cases by name. Returns the record.
"""
function run_benchmarks(; profile::Symbol = :small, save::Bool = true, only = nothing)
    haskey(PROFILES, profile) ||
        throw(ArgumentError("profile must be one of $(collect(keys(PROFILES))), got :$profile"))
    p = PROFILES[profile]
    started = now()
    results = Dict{String,Any}()
    mktempdir() do tmp
        with_output_root(tmp) do
            for (name, f) in benchmark_cases(p, tmp)
                isnothing(only) || occursin(only, name) || continue
                f()  # compile and populate caches outside the timed samples
                trial = @benchmark $f() evals = 1 samples = p.samples seconds = p.seconds
                results[name] = summarize(trial)
                @printf(
                    "%-34s %12s  %10d allocs  %12s\n",
                    name,
                    _format_ns(results[name]["median_ns"]),
                    results[name]["allocs"],
                    Base.format_bytes(results[name]["memory_bytes"])
                )
            end
        end
    end
    record = Dict{String,Any}(
        "profile" => String(profile),
        "timestamp" => string(started),
        "julia_version" => string(VERSION),
        "nthreads" => Threads.nthreads(),
        "cpu_model" => Sys.cpu_info()[1].model,
        "cases" => results,
    )
    if save
        name = savename(
            "benchmarks",
            (; profile = String(profile), time = Dates.format(started, "yyyymmdd-HHMMSS")),
            "jld2",
        )
        path = joinpath(benchmarks_dir(), name)
        mkpath(dirname(path))
        tagsave(path, record; safe = false)  # adds the git commit as "gitcommit"
        record["path"] = path
        println("saved ", relpath(path, projectdir()))
    end
    return record
end

"""
    latest_benchmarks(n = 2) -> Vector{String}

Paths of the `n` newest benchmark records, newest first. Also prints them with their profile
and timestamp.
"""
function latest_benchmarks(n::Integer = 2)
    dir = benchmarks_dir()
    files = isdir(dir) ? filter(f -> endswith(f, ".jld2"), readdir(dir; join = true)) : String[]
    sort!(files; by = mtime, rev = true)
    files = first(files, n)
    for f in files
        rec = JLD2.load(f)
        @printf("%s  (%s, %s)\n", relpath(f, projectdir()), rec["profile"], rec["timestamp"])
    end
    return files
end

"""
    compare_benchmarks(a, b)

Print a per-case table of median-time ratios `b / a` (below 1 means `b` is faster) for two
benchmark record paths. Cases present in only one record are skipped.
"""
function compare_benchmarks(a::AbstractString, b::AbstractString)
    ra, rb = JLD2.load(a), JLD2.load(b)
    for (label, r) in (("a", ra), ("b", rb))
        @printf(
            "%s: %s  %s  julia %s, %d threads, %s, commit %s\n",
            label,
            r["profile"],
            r["timestamp"],
            r["julia_version"],
            r["nthreads"],
            r["cpu_model"],
            get(r, "gitcommit", "?")
        )
    end
    ra["profile"] == rb["profile"] ||
        @warn "Records use different profiles; ratios are not comparable"
    @printf("\n%-34s %12s %12s %8s %8s\n", "case", "a", "b", "time", "memory")
    for name in sort!(collect(intersect(keys(ra["cases"]), keys(rb["cases"]))))
        ca, cb = ra["cases"][name], rb["cases"][name]
        @printf(
            "%-34s %12s %12s %7.2fx %7.2fx\n",
            name,
            _format_ns(ca["median_ns"]),
            _format_ns(cb["median_ns"]),
            cb["median_ns"] / ca["median_ns"],
            cb["memory_bytes"] / max(ca["memory_bytes"], 1)
        )
    end
    return nothing
end

function main()
    profile = isempty(ARGS) ? :small : Symbol(ARGS[1])
    run_benchmarks(; profile)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
