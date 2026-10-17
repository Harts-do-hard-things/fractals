# --------------------------------
# Fractal data records (JLD2 via DrWatson)
# --------------------------------
#
# A saved fractal is a `Dict{String,Any}` of plain arrays, strings and numbers (no Fractals
# types), so `JLD2.load(path)` / DrWatson's `wload(path)` reads it without this package:
#
#   "eq"        n×7 equation matrix (7th column: sampling weights)
#   "weights"   length-n weight vector          "limits"  2×2 [xmin xmax; ymin ymax]
#   "points"    2×N Matrix{Float64}             "image"   rasterized Matrix (optional)
#   "method"    render method name              "params"  Dict of parameters
#   "seed"      Int or nothing                  "meta"    small summary (read by the catalog)
#   "name", "format", "fractals_version", "created", and DrWatson's "gitcommit".
#
# Everything is written/read through the helpers in output.jl, into `data_output_dir()`.

const _FRACTAL_RECORD_FORMAT = 1

_points_matrix(points::AbstractVector{<:SVector{2,Float64}}) =
    Matrix{Float64}(reinterpret(reshape, Float64, points))

_points_vector(m::AbstractMatrix{<:Real}) =
    [SVector{2,Float64}(m[1, j], m[2, j]) for j in axes(m, 2)]

function _equation_matrix(ifs::IFS)
    n = length(ifs.maps)
    eq = Matrix{Float64}(undef, n, 7)
    for (i, m) in enumerate(ifs.maps)
        eq[i, :] .= (m.A[1, 1], m.A[1, 2], m.A[2, 1], m.A[2, 2], m.b[1], m.b[2], ifs.weights[i])
    end
    return eq
end

_eq_hash(eq::AbstractMatrix) = string(hash(Float64.(eq)); base = 16, pad = 16)[1:8]

_record_params(params::NamedTuple) =
    Dict{String,Any}(String(k) => _provenance_value(v) for (k, v) in pairs(params))

"""
    fractal_record(ifs, points; image=nothing, method=:chaos, params=(;), seed=nothing) -> Dict{String,Any}

The plain-data record that [`save_fractal`](@ref) writes for `ifs` and its `points`: the
equation matrix, weights, limits, points (a `2×N` matrix), optional `image`, render `method`, `params`,
`seed`, and a `"meta"` summary. Contains only arrays, strings and numbers.
"""
function fractal_record(
    ifs::IFS,
    points::AbstractVector{<:SVector{2,Float64}};
    image = nothing,
    method::Union{Symbol,AbstractString} = :chaos,
    params::NamedTuple = (;),
    seed::Union{Nothing,Integer} = nothing,
)
    created = string(now())
    record = Dict{String,Any}(
        "format" => _FRACTAL_RECORD_FORMAT,
        "fractals_version" => string(pkgversion(@__MODULE__)),
        "created" => created,
        "name" => ifs.name,
        "eq" => _equation_matrix(ifs),
        "weights" => Float64.(collect(ifs.weights)),
        "limits" =>
            Float64[ifs.limits[1][1] ifs.limits[1][2]; ifs.limits[2][1] ifs.limits[2][2]],
        "points" => _points_matrix(points),
        "method" => String(method),
        "params" => _record_params(params),
        "seed" => isnothing(seed) ? nothing : Int(seed),
        "meta" => Dict{String,Any}(
            "name" => ifs.name,
            "method" => String(method),
            "npoints" => length(points),
            "nmaps" => length(ifs.maps),
            "seed" => isnothing(seed) ? nothing : Int(seed),
            "created" => created,
            "has_image" => !isnothing(image),
        ),
    )
    isnothing(image) || (record["image"] = image)
    return record
end

function _ifs_from_record(record::AbstractDict)
    eq = record["eq"]
    maps, weights = _build_maps_and_weights(eq)
    lim = record["limits"]
    limits = ((Float64(lim[1, 1]), Float64(lim[1, 2])), (Float64(lim[2, 1]), Float64(lim[2, 2])))
    return IFS(record["name"], maps, weights, limits)
end

function _default_data_name(ifs::IFS, npoints, method, seed)
    return _output_name(ifs.name, (; method = Symbol(method), npoints, seed), "jld2")
end

"""
    save_fractal(ifs, points; image=nothing, method=:chaos, params=(;), seed=nothing, outpath=nothing) -> path

Save `ifs` (maps, weights, limits) and its `points` as a JLD2 record, plus an
optional rasterized `image`, the render `method`, `params` and `seed`. `outpath=nothing`
names the file with DrWatson's `savename` under [`data_output_dir`](@ref); relative paths
resolve there and absolute paths are used as given. Saved with `safesave` (existing files
become `_#N` backups) and tagged with the Git commit like DrWatson's `tagsave` does. Read back with
[`load_fractal`](@ref), or with plain `JLD2.load` / `DrWatson.wload`.

```julia
ifs = IFS(EISENSTEIN); points = chaos_game(ifs, 100_000; seed = 1)
path = save_fractal(ifs, points; seed = 1, image = make_image(points, ifs.limits))
```
"""
function save_fractal(
    ifs::IFS,
    points::AbstractVector{<:SVector{2,Float64}};
    image = nothing,
    method::Union{Symbol,AbstractString} = :chaos,
    params::NamedTuple = (;),
    seed::Union{Nothing,Integer} = nothing,
    outpath::Union{Nothing,AbstractString} = nothing,
)
    path = _resolve_data_path(outpath, _default_data_name(ifs, length(points), method, seed))
    record = fractal_record(ifs, points; image, method, params, seed)
    return _save_data_record(path, record)
end

"""
    load_fractal(path) -> (ifs, points, image, method, params, seed, gitcommit, created, path, record)

Load a record written by [`save_fractal`](@ref) (via DrWatson's `wload`). `ifs` is the
[`IFS`](@ref) with its limits restored, `points` the saved point cloud, `image` is the saved raster or `nothing`,
`method` a `Symbol`, and `record` the raw `Dict`. Relative paths resolve under
[`data_output_dir`](@ref).
"""
function load_fractal(path::AbstractString)
    final_path = _resolve_data_path(path, "")
    record = _load_data_record(final_path)
    get(record, "format", nothing) == _FRACTAL_RECORD_FORMAT ||
        throw(ArgumentError("'$final_path' is not a Fractals data record"))
    return (
        ifs = _ifs_from_record(record),
        points = _points_vector(record["points"]),
        image = get(record, "image", nothing),
        method = Symbol(record["method"]),
        params = record["params"],
        seed = record["seed"],
        gitcommit = get(record, "gitcommit", nothing),
        created = record["created"],
        path = final_path,
        record = record,
    )
end

_loaded_fractal(f::AbstractString) = load_fractal(f)
_loaded_fractal(f::NamedTuple) = load_fractal(f.path) # e.g. a `fractal_data` or `load_fractal` result

"""
    fractal_data(input; method=:chaos, npoints=DEFAULT_SAMPLES, warmup=DEFAULT_WARMUP,
                 iterations=1, seed=0, name="", force=false) -> (ifs, points, path, cached, record)

Compute a fractal's point cloud once and cache it with DrWatson's `produce_or_load`. `input`
is an equation matrix or an [`IFS`](@ref); `method` is `:chaos` ([`chaos_game`](@ref)) or
`:point_deterministic` (a chaos-game seed expanded with [`deterministic_iterate`](@ref) to depth `iterations`). The file
name comes from `savename` over the equation matrix (hashed), method and parameters, so a
later call with identical arguments reloads the saved points (`cached = true`) instead of
recomputing them. `force=true` recomputes. The `seed` makes chaos-game output reproducible
for a fixed thread count.

Rasterize the result at any resolution with [`render_fractal`](@ref), without recomputing.
"""
function fractal_data(
    input::Union{AbstractMatrix{<:Real},IFS};
    method::Union{Symbol,AbstractString} = :chaos,
    npoints::Integer = DEFAULT_SAMPLES,
    warmup::Integer = DEFAULT_WARMUP,
    iterations::Integer = 1,
    seed::Integer = 0,
    name::AbstractString = "",
    force::Bool = false,
)
    m = Symbol(method)
    m in (:chaos, :point_deterministic) ||
        throw(ArgumentError("Invalid method '$method'. Supported: :chaos, :point_deterministic"))
    npoints > 0 || throw(ArgumentError("npoints must be > 0, got $npoints"))
    iterations >= 0 || throw(ArgumentError("iterations must be >= 0, got $iterations"))

    base = input isa IFS ? input : IFS(input; name = name)
    base_eq = _equation_matrix(base)
    label = isempty(name) ? base.name : String(name)
    config = (;
        method = m,
        npoints,
        warmup,
        iterations = m == :chaos ? nothing : iterations,
        seed,
        eq = _eq_hash(base_eq),
    )
    filename = _output_name(label, config, "jld2")

    function produce(_)
        out = IFS(label, base.maps, base.weights, base.limits)
        points = chaos_game(out, npoints; warmup, seed)
        m == :point_deterministic && (points = deterministic_iterate(out, points, iterations))
        params = (; npoints, warmup, iterations = config.iterations)
        return fractal_record(out, points; method = m, params, seed)
    end

    record, path, cached = _produce_or_load_record(produce, config, filename; force)
    return (
        ifs = _ifs_from_record(record),
        points = _points_vector(record["points"]),
        path = path,
        cached = cached,
        record = record,
    )
end

"""
    render_fractal(source; resolution=RESOLUTION, color=false, alpha=false, backend=:cpu,
                   outpath=nothing) -> (ifs, image, outpath, method)

Rasterize saved fractal data to a PNG without recomputing points. `source` is a path to a
JLD2 record, or the result of [`load_fractal`](@ref) or [`fractal_data`](@ref). Resolution,
coloring and transparency can differ from when the data was computed. The PNG is saved like
[`render`](@ref) output (same naming and provenance).
"""
function render_fractal(
    source;
    resolution::Tuple{Int,Int} = RESOLUTION,
    color::Bool = false,
    alpha::Bool = false,
    backend::Symbol = :cpu,
    outpath::Union{Nothing,AbstractString} = nothing,
)
    resolution[1] > 0 && resolution[2] > 0 ||
        throw(ArgumentError("resolution must be positive, got $resolution"))
    loaded = _loaded_fractal(source)
    ifs = loaded.ifs
    method = loaded.method
    points = loaded.points
    img = make_image(points, ifs.limits; resolution = resolution, backend = backend)
    color && (img = iterate_image(ifs, img; colors = true, backend = backend))
    alpha && (img = _apply_alpha_mask(img))
    final_outpath = _save_render_output(
        outpath,
        img,
        ifs,
        ifs.limits;
        method = method,
        params = (; npoints = length(points), resolution, color, alpha, backend),
        name_keys = (:npoints, :resolution, :color, :alpha),
    )
    return (ifs = ifs, image = img, outpath = final_outpath, method = method)
end

"""
    extend_fractal(source, n; seed=nothing, outpath=nothing) -> (ifs, points, path)

Resume a saved chaos game: load `source` (a path or [`load_fractal`](@ref) result), continue
the chain from its last point for `n` more steps, and save the accumulated cloud as a new
record (named for the new point count, so the earlier file is kept). Repeated calls keep
refining the same fractal. The saved raster, if any, is dropped as stale. Only records with
method `:chaos` can be extended. `seed` makes the extra steps reproducible.
"""
function extend_fractal(
    source,
    n::Integer;
    seed::Union{Nothing,Integer} = nothing,
    outpath::Union{Nothing,AbstractString} = nothing,
)
    n > 0 || throw(ArgumentError("n must be > 0, got $n"))
    loaded = _loaded_fractal(source)
    loaded.method == :chaos ||
        throw(ArgumentError("Only :chaos records can be extended, got :$(loaded.method)"))
    old = loaded.ifs
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    indices = Base.OneTo(length(old.maps))
    x = isempty(loaded.points) ? SVector{2,Float64}(0.0, 0.0) : last(loaded.points)
    points = copy(loaded.points)
    sizehint!(points, length(points) + n)
    for _ = 1:n
        x = old.maps[sample(rng, indices, old.weights)](x)
        push!(points, x)
    end
    params = merge(
        (; (Symbol(k) => v for (k, v) in loaded.params)...),
        (; npoints = length(points), extended_by = n),
    )
    path = save_fractal(
        old,
        points;
        method = :chaos,
        params = params,
        seed = loaded.seed,
        outpath = outpath,
    )
    return (ifs = old, points = points, path = path)
end

"""
    fractal_catalog(dir=data_output_dir(); recursive=true) -> Vector{NamedTuple}

Catalog of the saved fractals under `dir`: one `(path, name, method, npoints, nmaps, seed,
created, has_image, gitcommit)` row per JLD2 record written by [`save_fractal`](@ref) or
[`fractal_data`](@ref), oldest first. Only each record's small summary is read, not its
points, so cataloging a large data directory is cheap. The rows are plain named tuples, so
they work directly with `DataFrame(rows)` if DataFrames is loaded.
"""
function fractal_catalog(dir::AbstractString = data_output_dir(); recursive::Bool = true)
    rows = NamedTuple[]
    isdir(dir) || return rows
    paths = String[]
    for (root, _, files) in walkdir(dir)
        for f in files
            endswith(f, ".jld2") && push!(paths, joinpath(root, f))
        end
        recursive || break
    end
    for path in sort!(paths)
        row = try
            JLD2.jldopen(path, "r") do file
                haskey(file, "meta") || return nothing
                meta = file["meta"]
                (;
                    path,
                    name = meta["name"],
                    method = Symbol(meta["method"]),
                    npoints = meta["npoints"],
                    nmaps = meta["nmaps"],
                    seed = meta["seed"],
                    created = meta["created"],
                    has_image = meta["has_image"],
                    gitcommit = haskey(file, "gitcommit") ? file["gitcommit"] : nothing,
                )
            end
        catch
            nothing # not a readable JLD2 file
        end
        isnothing(row) || push!(rows, row)
    end
    return sort!(rows; by = r -> r.created)
end
