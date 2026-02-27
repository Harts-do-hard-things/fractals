module BenchmarkSuite

using Fractals
using Dates
using JSON3
using Statistics
using TOML

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
    allocs = Int[]
    for _ in 1:repeats
        t = @timed f()
        push!(times, t.time)
        push!(allocs, t.bytes)
    end
    return (
        min=minimum(times),
        mean=mean(times),
        max=maximum(times),
        min_alloc=minimum(allocs),
        mean_alloc=mean(allocs),
        max_alloc=maximum(allocs),
        samples=times,
    )
end

function _metric_dict(stats)
    return Dict(
        "min_s" => stats.min,
        "mean_s" => stats.mean,
        "max_s" => stats.max,
        "min_alloc_bytes" => stats.min_alloc,
        "mean_alloc_bytes" => stats.mean_alloc,
        "max_alloc_bytes" => stats.max_alloc,
    )
end

function _hardware_info()
    return Dict(
        "cpu" => Sys.CPU_NAME,
        "os" => string(Sys.KERNEL),
        "arch" => string(Sys.ARCH),
        "threads" => Threads.nthreads(),
        "total_memory_bytes" => Sys.total_memory(),
    )
end

function _bench_one(label::String, npoints::Int, resolution::Tuple{Int,Int}, inverse_iterations::Int, repeats::Int)
    warmup = DEFAULT_WARMUP
    ifs = IFS(HEIGHWAY_DRAGON; npoints=npoints)
    # Warmup for JIT and method compilation.
    warm_ifs = IFS(HEIGHWAY_DRAGON; npoints=5_000)
    iterate!(warm_ifs; warmup=10, seed=1234)
    iterate_parallel!(IFS(HEIGHWAY_DRAGON; npoints=5_000); warmup=10, seed=1234)
    make_image(warm_ifs; resolution=(64, 64))
    rasterize_image_inversely(warm_ifs, 1, warm_ifs.limits; resolution=(32, 32))

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
        "iterate!" => _metric_dict(iter_stats),
        "iterate_parallel!" => _metric_dict(iter_parallel_stats),
        "make_image" => _metric_dict(image_stats),
        "rasterize_image_inversely" => _metric_dict(inverse_stats),
    )
end

function _target_metric_status(actual::Float64, target::Float64, warn_ratio::Float64, fail_ratio::Float64)
    ratio = target > 0 ? actual / target : 1.0
    status = ratio <= warn_ratio ? "pass" : (ratio <= fail_ratio ? "warn" : "fail")
    return Dict(
        "actual" => actual,
        "target" => target,
        "ratio" => ratio,
        "status" => status,
    )
end

function _worst_status(a::String, b::String)
    rank = Dict("pass" => 1, "warn" => 2, "fail" => 3)
    return rank[a] >= rank[b] ? a : b
end

function _compare_against_targets(payload::Dict{String,Any}, targets_path::String)
    raw = TOML.parsefile(targets_path)
    thresholds = get(raw, "thresholds", Dict{String,Any}())
    warn_ratio = Float64(get(thresholds, "warn_ratio", 1.15))
    fail_ratio = Float64(get(thresholds, "fail_ratio", 1.30))

    all_targets = get(raw, "targets", Dict{String,Any}())
    comparisons = Dict{String,Any}()
    overall_status = "pass"

    for (profile_name, result) in payload["results"]
        profile_targets = get(all_targets, profile_name, Dict{String,Any}())
        ops = Dict{String,Any}()
        profile_status = "pass"

        for op in ("iterate!", "iterate_parallel!", "make_image", "rasterize_image_inversely")
            if !haskey(profile_targets, op) || !haskey(result, op)
                continue
            end
            op_target = profile_targets[op]
            op_result = result[op]

            mean_s = _target_metric_status(
                Float64(op_result["mean_s"]),
                Float64(get(op_target, "mean_s", 0.0)),
                warn_ratio,
                fail_ratio,
            )
            mean_alloc = _target_metric_status(
                Float64(op_result["mean_alloc_bytes"]),
                Float64(get(op_target, "mean_alloc_bytes", 0.0)),
                warn_ratio,
                fail_ratio,
            )

            op_status = _worst_status(mean_s["status"], mean_alloc["status"])
            profile_status = _worst_status(profile_status, op_status)
            ops[op] = Dict(
                "mean_s" => mean_s,
                "mean_alloc_bytes" => mean_alloc,
                "status" => op_status,
            )
        end

        overall_status = _worst_status(overall_status, profile_status)
        comparisons[profile_name] = Dict("status" => profile_status, "ops" => ops)
    end

    return Dict(
        "targets_path" => targets_path,
        "warn_ratio" => warn_ratio,
        "fail_ratio" => fail_ratio,
        "status" => overall_status,
        "profiles" => comparisons,
    )
end

function run_suite(; profile::String="small",
                     repeats::Int=3,
                     json_path::Union{Nothing,String}=nothing,
                     targets_path::Union{Nothing,String}=nothing,
                     strict::Bool=false)
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
        "hardware" => _hardware_info(),
        "repeats" => repeats,
        "results" => results,
    )

    if !isnothing(targets_path)
        comparison = _compare_against_targets(payload, targets_path)
        payload["comparison"] = comparison
        if strict && comparison["status"] == "fail"
            throw(ErrorException("Benchmark targets failed in strict mode (status=fail)."))
        end
    end

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
    if haskey(payload, "hardware")
        hw = payload["hardware"]
        println("  cpu:       ", hw["cpu"])
        println("  os/arch:   ", hw["os"], "/", hw["arch"])
    end
    println("")

    for name in sort(collect(keys(payload["results"])))
        item = payload["results"][name]
        println("[$name] npoints=$(item["npoints"]) resolution=$(Tuple(item["resolution"])) inverse_iterations=$(item["inverse_iterations"])")
        for key in ("iterate!", "iterate_parallel!", "make_image", "rasterize_image_inversely")
            stats = item[key]
            println("  ", rpad(key, 26),
                    " min=$(round(stats["min_s"], digits=4))s  mean=$(round(stats["mean_s"], digits=4))s  max=$(round(stats["max_s"], digits=4))s",
                    "  alloc_mean=$(round(stats["mean_alloc_bytes"] / 1024^2, digits=3)) MiB")
        end
        println("")
    end

    if haskey(payload, "comparison")
        cmp = payload["comparison"]
        println("Target comparison")
        println("  targets:   ", cmp["targets_path"])
        println("  status:    ", cmp["status"])
        for name in sort(collect(keys(cmp["profiles"])))
            prof = cmp["profiles"][name]
            println("  [$name] status=", prof["status"])
            for op in ("iterate!", "iterate_parallel!", "make_image", "rasterize_image_inversely")
                haskey(prof["ops"], op) || continue
                opcmp = prof["ops"][op]
                ts = opcmp["mean_s"]
                ta = opcmp["mean_alloc_bytes"]
                println("    ", rpad(op, 24),
                        " time=", opcmp["status"],
                        " (", round(ts["actual"], digits=4), "s vs ", round(ts["target"], digits=4), "s, x", round(ts["ratio"], digits=2), ")",
                        "  alloc=", ta["status"],
                        " (", round(ta["actual"] / 1024^2, digits=3), " MiB vs ",
                        round(ta["target"] / 1024^2, digits=3), " MiB, x", round(ta["ratio"], digits=2), ")")
            end
        end
    end
end

end
