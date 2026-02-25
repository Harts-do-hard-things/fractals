module BenchmarkSuite

using Fractals
using Dates
using JSON3
using Statistics

function _profile_spec(profile::String)
    p = lowercase(strip(profile))
    p in ("small", "medium", "large", "all") || throw(ArgumentError("Invalid profile '$profile'. Expected small|medium|large|all"))

    small = (
        npoints=20_000,
        resolution=(128, 128),
        inverse_iterations=2,
    )
    medium = (
        npoints=100_000,
        resolution=(256, 256),
        inverse_iterations=3,
    )
    large = (
        npoints=300_000,
        resolution=(512, 512),
        inverse_iterations=4,
    )

    if p == "small"
        return Dict("small" => small)
    elseif p == "medium"
        return Dict("medium" => medium)
    elseif p == "large"
        return Dict("large" => large)
    end
    return Dict("small" => small, "medium" => medium, "large" => large)
end

function _time_repeats(f::Function, repeats::Int)
    times = Float64[]
    for _ in 1:repeats
        push!(times, @elapsed f())
    end
    return (
        min=minimum(times),
        mean=mean(times),
        max=maximum(times),
        samples=times,
    )
end

function _bench_one(label::String, npoints::Int, resolution::Tuple{Int,Int}, inverse_iterations::Int, repeats::Int)
    warmup = DEFAULT_WARMUP
    ifs = IFS(HEIGHWAY_DRAGON; npoints=npoints)
    # Warmup for JIT and method compilation.
    iterate!(IFS(HEIGHWAY_DRAGON; npoints=5_000); warmup=10, seed=1234)

    iter_stats = _time_repeats(repeats) do
        local tifs = IFS(HEIGHWAY_DRAGON; npoints=npoints)
        iterate!(tifs; warmup=warmup, seed=1234)
    end

    iter_parallel_stats = _time_repeats(repeats) do
        local tifs = IFS(HEIGHWAY_DRAGON; npoints=npoints)
        iterate_parallel!(tifs; warmup=warmup, seed=1234)
    end

    # Build once for make_image benchmark.
    iterate!(ifs; warmup=warmup, seed=1234)
    image_stats = _time_repeats(repeats) do
        make_image(ifs; resolution=resolution)
    end

    inverse_stats = _time_repeats(repeats) do
        rasterize_image_inversely(ifs, inverse_iterations, ifs.limits; resolution=resolution)
    end

    return Dict(
        "profile" => label,
        "npoints" => npoints,
        "resolution" => [resolution[1], resolution[2]],
        "inverse_iterations" => inverse_iterations,
        "warmup" => warmup,
        "threads" => Threads.nthreads(),
        "iterate!" => Dict("min_s" => iter_stats.min, "mean_s" => iter_stats.mean, "max_s" => iter_stats.max),
        "iterate_parallel!" => Dict("min_s" => iter_parallel_stats.min, "mean_s" => iter_parallel_stats.mean, "max_s" => iter_parallel_stats.max),
        "make_image" => Dict("min_s" => image_stats.min, "mean_s" => image_stats.mean, "max_s" => image_stats.max),
        "rasterize_image_inversely" => Dict("min_s" => inverse_stats.min, "mean_s" => inverse_stats.mean, "max_s" => inverse_stats.max),
    )
end

function run_suite(; profile::String="small", repeats::Int=3, json_path::Union{Nothing,String}=nothing)
    repeats > 0 || throw(ArgumentError("repeats must be > 0, got $repeats"))

    specs = _profile_spec(profile)
    results = Dict{String,Any}()

    for (name, spec) in specs
        results[name] = _bench_one(name, spec.npoints, spec.resolution, spec.inverse_iterations, repeats)
    end

    payload = Dict(
        "timestamp" => string(now(UTC)),
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "repeats" => repeats,
        "results" => results,
    )

    if !isnothing(json_path)
        write(json_path, JSON3.write(payload))
    end

    return payload
end

function print_report(payload::Dict{String,Any})
    println("Benchmark suite")
    println("  timestamp: ", payload["timestamp"])
    println("  julia:     ", payload["julia_version"])
    println("  threads:   ", payload["threads"])
    println("  repeats:   ", payload["repeats"])
    println("")

    for name in sort(collect(keys(payload["results"])))
        item = payload["results"][name]
        println("[$name] npoints=$(item["npoints"]) resolution=$(Tuple(item["resolution"])) inverse_iterations=$(item["inverse_iterations"])")
        for key in ("iterate!", "iterate_parallel!", "make_image", "rasterize_image_inversely")
            stats = item[key]
            println("  ", rpad(key, 26), " min=$(round(stats["min_s"], digits=4))s  mean=$(round(stats["mean_s"], digits=4))s  max=$(round(stats["max_s"], digits=4))s")
        end
        println("")
    end
end

end
