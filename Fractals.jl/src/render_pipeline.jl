# --------------------------------
# Render pipeline
# --------------------------------
#
# One pipeline for every method (`render(::Render)`); everything method-specific is a method of
# `draw`, `colorize`, `seed_image`, or `overlay!`. Each `draw` returns `(image, limits)`, where
# `limits` is the world-space frame of that image (it differs from `ifs.limits` for polygon-
# framed methods). Must be included after render_types.jl and every file whose kernels it calls.

_with_limits(ifs::IFS, limits) = IFS(ifs.name, ifs.maps, ifs.weights, limits)

# The raster helpers in drawing/ only read `.limits` from their `ifs` argument.
_frame(limits) = (; limits)

# --------------------------------
# Point clouds: computed once, cached independently of resolution and styling
# --------------------------------
#
# The point-based methods split into an expensive step (the point cloud, which depends only on
# the IFS's maps and weights and the method's fields) and a cheap one (rasterizing at
# `resolution`). The cloud is cached with DrWatson's `produce_or_load` as one flat file per cloud
# in `data_output_dir("points")`, named by a config dictionary holding the IFS content key and
# every method field (e.g. `chaos_ifs=<key>_npoints=1000_seed=1_warmup=50.jld2`). The key ignores
# the IFS name, resolution, color, alpha and overlays, so re-rendering at another size or style
# reuses the same points.

_points_matrix(points::AbstractVector{<:SVector{2,Float64}}) =
    Matrix{Float64}(reinterpret(reshape, Float64, points))

_points_vector(m::AbstractMatrix{<:Real}) =
    [SVector{2,Float64}(m[1, j], m[2, j]) for j in axes(m, 2)]

# The n×7 equation matrix (`a11 a12 a21 a22 b1 b2 p`) of `ifs`.
function _equation_matrix(ifs::IFS)
    n = length(ifs.maps)
    eq = Matrix{Float64}(undef, n, 7)
    for (i, m) in enumerate(ifs.maps)
        eq[i, :] .= (m.A[1, 1], m.A[1, 2], m.A[2, 1], m.A[2, 2], m.b[1], m.b[2], ifs.weights[i])
    end
    return eq
end

"""
    ifs_key(ifs) -> String

Stable content key of `ifs`: a hash of its maps and weights only (not its name), the same hash
IFSParse uses for prebuilt library records.
"""
ifs_key(ifs::IFS) = IFSParse._ifs_key(_equation_matrix(ifs), collect(ifs.weights))

_compute_points(m::Chaos, ifs::IFS) = chaos_game(ifs, m.npoints; warmup = m.warmup, seed = m.seed)
_compute_points(m::PointDeterministic, ifs::IFS) = deterministic_iterate(
    ifs,
    chaos_game(ifs, m.npoints; warmup = m.warmup, seed = m.seed),
    m.iterations,
)

# DrWatson config for a cached cloud: the IFS key plus every method field (`nothing` fields, such
# as an unset seed, are left out of the name by `savename`).
_points_config(m::RenderMethod, ifs::IFS) = Dict{String,Any}(
    "ifs" => ifs_key(ifs),
    (String(f) => getfield(m, f) for f in fieldnames(typeof(m)))...,
)

"""
    cached_points(method, ifs; force = false) -> (points, path, cached)

The point cloud of a [`Chaos`](@ref) or [`PointDeterministic`](@ref) `method` for `ifs`, computed
once and then loaded from `data_output_dir("points")`. Each cloud is its own file, named by
DrWatson's `savename` over the IFS content key ([`ifs_key`](@ref)) and every field of `method`,
e.g. `chaos_ifs=<key>_npoints=1000_seed=1_warmup=50.jld2`. Resolution and styling are not part
of the key. An unseeded method is cached too, so repeated renders reuse one cloud;
`force = true` recomputes. Grow a cached chaos game with [`extend_fractal`](@ref).
"""
function cached_points(m::Union{Chaos,PointDeterministic}, ifs::IFS; force::Bool = false)
    record, path, cached = _points_entry(() -> _compute_points(m, ifs), m, ifs; force)
    return _points_vector(record["points"]), path, cached
end

# Produce-or-load one cloud entry. The record holds `points` (a 2×N matrix), the `method`, the
# IFS content key as `ifs`, and DrWatson's `gitcommit`. Returns `(record, path, cached)`.
function _points_entry(produce::Function, m::RenderMethod, ifs::IFS; force::Bool)
    config = _points_config(m, ifs)
    computed = Ref(false)
    record, path = DrWatson.produce_or_load(
        config,
        data_output_dir("points");
        filename = savename(method_tag(m), config),
        suffix = "jld2",
        tag = false,
        force = force,
        verbose = false,
    ) do _
        computed[] = true
        _tag_record!(
            Dict{String,Any}(
                "points" => _points_matrix(produce()),
                "method" => m,
                "ifs" => config["ifs"],
            ),
        )
    end
    return record, path, !computed[]
end

"""
    extend_fractal(method::Chaos, ifs::IFS, n::Integer; seed = nothing) -> (points, path)

Continue the chaos game cached by [`cached_points`](@ref)`(method, ifs)` from its last point for
`n` more steps (maps drawn by `ifs.weights`, as in [`chaos_game`](@ref)), and store the
accumulated cloud as the cache entry of `Chaos(npoints = method.npoints + n, warmup = method.warmup,
seed = method.seed)`. A later `render(ifs, Chaos(npoints = method.npoints + n, ...))` therefore
reuses it. An existing entry for the new count is overwritten. `seed` makes the extra steps
reproducible; the original cloud is left in place. Returns the new cloud and its path.
"""
function extend_fractal(method::Chaos, ifs::IFS, n::Integer; seed::Union{Nothing,Integer} = nothing)
    n > 0 || throw(ArgumentError("n must be > 0, got $n"))
    old, _, _ = cached_points(method, ifs)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    indices = Base.OneTo(length(ifs.maps))
    x = isempty(old) ? SVector{2,Float64}(0.0, 0.0) : last(old)
    points = copy(old)
    sizehint!(points, length(points) + n)
    for _ = 1:n
        x = ifs.maps[sample(rng, indices, ifs.weights)](x)
        push!(points, x)
    end
    extended = Chaos(; npoints = method.npoints + n, warmup = method.warmup, seed = method.seed)
    _, path, _ = _points_entry(() -> points, extended, ifs; force = true)
    return points, path
end

"""
    fractal_catalog(; dir = data_output_dir()) -> Vector{NamedTuple}

List the saved renders (`dir/renders`) and cached point clouds (`dir/points`) as rows
`(kind, name, method, npoints, resolution, ifs_key, gitcommit, path)`, sorted by `path`.
`kind` is `:render` or `:points`; `name` is the IFS name (`missing` for point clouds, which do
not depend on it); `method` is the method tag; `npoints` is `missing` for methods without it;
`resolution` is `missing` for point clouds; `gitcommit` is `nothing` when unrecorded. Only
small keys of each JLD2 file are read, never the points, and unreadable files are skipped.
The rows work directly with `DataFrame(rows)`.
"""
function fractal_catalog(; dir::AbstractString = data_output_dir())
    rows = NamedTuple[]
    for (kind, sub) in ((:render, "renders"), (:points, "points"))
        root = joinpath(dir, sub)
        isdir(root) || continue
        for (folder, _, files) in walkdir(root), f in files
            endswith(f, ".jld2") || continue
            path = joinpath(folder, f)
            row = try
                _catalog_row(kind, path)
            catch
                nothing # not a readable record
            end
            isnothing(row) || push!(rows, row)
        end
    end
    return sort!(rows; by = r -> r.path)
end

_npoints(m::RenderMethod) = hasfield(typeof(m), :npoints) ? m.npoints : missing

function _catalog_row(kind::Symbol, path::AbstractString)
    return JLD2.jldopen(path, "r") do file
        gitcommit = haskey(file, "gitcommit") ? file["gitcommit"] : nothing
        if kind === :render
            r = file["render"]::Render
            (;
                kind,
                name = r.ifs.name,
                method = method_tag(r.method),
                npoints = _npoints(r.method),
                resolution = r.resolution,
                ifs_key = ifs_key(r.ifs),
                gitcommit,
                path,
            )
        else
            m = file["method"]::RenderMethod
            (;
                kind,
                name = missing,
                method = method_tag(m),
                npoints = _npoints(m),
                resolution = missing,
                ifs_key = file["ifs"],
                gitcommit,
                path,
            )
        end
    end
end

# --------------------------------
# draw: one method per render method
# --------------------------------

draw(r::Render) = draw(r.method, r)

function draw(m::Union{Chaos,PointDeterministic}, r::Render)
    points, _, _ = cached_points(m, r.ifs)
    return make_image(points, r.ifs.limits; resolution = r.resolution), r.ifs.limits
end

function draw(m::Inverse, r::Render)
    img = rasterize_image_inversely(
        r.ifs,
        m.iterations,
        r.ifs.limits;
        resolution = r.resolution,
        show_divergence_scale = m.divergence_scale,
        mode = m.preview ? :preview : :exact,
    )
    return img, r.ifs.limits
end

function draw(m::Transformations, r::Render)
    rows, cols = r.resolution
    img = _render_transformations_image(
        r.ifs;
        width = cols,
        height = rows,
        initial_polygon_spec = m.polygon,
        polygon_limits_iterations = m.limits_iterations,
        color = r.color,
    )
    limits = _limits_from_polygon_point_iterations(
        r.ifs;
        initial_polygon_spec = m.polygon,
        polygon_limits_iterations = m.limits_iterations,
    )
    return img, limits
end

function draw(m::ImageIterate, r::Render)
    img, limits = seed_image(m.source, r)
    raster_ifs = _with_limits(r.ifs, limits)
    for pass = 1:m.iterations
        img = iterate_image(raster_ifs, img; colors = r.color && pass == m.iterations)
    end
    return img, limits
end

# --------------------------------
# seed_image: where ImageIterate starts
# --------------------------------

function seed_image(s::PolygonSource, r::Render)
    limits =
        s.frame == :ifs ? r.ifs.limits :
        _limits_from_polygon_point_iterations(
            r.ifs;
            initial_polygon_spec = s.polygon,
            polygon_limits_iterations = something(s.limits_iterations, r.method.iterations),
        )
    rows, cols = r.resolution
    img = _render_initial_polygon_seed_image(
        _frame(limits);
        width = cols,
        height = rows,
        initial_polygon_spec = s.polygon,
    )
    return _to_grayscale_matrix(img), limits
end

function seed_image(s::FileSource, r::Render)
    img = _load_image(s.path)
    _validate_iterate_image_input(img)
    record = render_record_path(s.path)
    limits = isfile(record) ? JLD2.load(record, "limits") : r.ifs.limits
    return _to_grayscale_matrix(img), limits
end

# Any render method can seed ImageIterate: its image (and frame) is the starting image.
function seed_image(m::RenderMethod, r::Render)
    img, limits = draw(m, r)
    return _to_grayscale_matrix(img), limits
end

# --------------------------------
# Finishing: color, overlays, alpha
# --------------------------------

# Default coloring is one colored iterate_image pass. Methods that color while drawing opt out.
colorize(::RenderMethod, r::Render, img, limits) =
    iterate_image(_with_limits(r.ifs, limits), img; colors = true)
colorize(::Transformations, ::Render, img, limits) = img
colorize(::ImageIterate, ::Render, img, limits) = img

function overlay!(img, o::BasePolygon, r::Render, limits)
    rows, cols = r.resolution
    return _overlay_initial_polygon!(
        img,
        _frame(limits);
        width = cols,
        height = rows,
        initial_polygon_spec = o.polygon,
        color = r.color,
    )
end

function overlay!(img, ::Axes, r::Render, limits)
    rows, cols = r.resolution
    return _overlay_axes!(img, _frame(limits); width = cols, height = rows)
end

function finish(r::Render, img, limits)
    r.color && (img = colorize(r.method, r, img, limits))
    # Overlay helpers draw into Gray/RGB pixels; raw heatmaps are plain Float32 matrices.
    isempty(r.overlays) || eltype(img) <: Real && (img = Gray{Float32}.(img))
    for o in r.overlays
        overlay!(img, o, r, limits)
    end
    r.alpha && (img = _apply_alpha_mask(img))
    return img
end

# --------------------------------
# Naming, saving, loading (DrWatson + JLD2)
# --------------------------------

# Which method fields go into file names (DrWatson's `savename` reads these).
DrWatson.allaccess(::Chaos) = (:npoints, :seed)
DrWatson.allaccess(::PointDeterministic) = (:npoints, :iterations)
# `preview` only appears in names when set, so exact renders keep their names.
DrWatson.allaccess(m::Inverse) = m.preview ? (:iterations, :preview) : (:iterations,)
DrWatson.allaccess(::Transformations) = (:polygon, :limits_iterations)
DrWatson.allaccess(::ImageIterate) = (:iterations,)

method_tag(::Chaos) = "chaos"
method_tag(::PointDeterministic) = "point_deterministic"
method_tag(::Inverse) = "inverse"
method_tag(::Transformations) = "transformations"
method_tag(m::ImageIterate) = "image_iterate-" * source_tag(m.source)
source_tag(::PolygonSource) = "polygon"
source_tag(::FileSource) = "file"
source_tag(m::RenderMethod) = method_tag(m)

function _render_name(r::Render)
    stem = savename(_savename_prefix(r.ifs.name) * "_" * method_tag(r.method), r.method)
    rows, cols = r.resolution
    flags = (r.color ? "_color" : "") * (r.alpha ? "_alpha" : "")
    return "$(stem)_$(rows)x$(cols)$(flags).png"
end

"""
    render_record_path(r::Render) -> String
    render_record_path(image_path) -> String

Where a render record lives. Records always go under `data_output_dir("renders")` and are
never placed by an `outpath`:

- `render_record_path(r)`: named from the render itself (the same `savename`-based name as its
  default PNG), whatever `outpath` the PNG was written to.
- `render_record_path(image_path)`: the record that mirrors an image under `output_dir()`
  (`plots/<rel>.png` → `renders/<rel>.jld2`). This finds the record of a PNG at its default
  location and of animation frames, which keep their `frames/<sequence>/` subdirectories. A PNG
  written to a custom `outpath` has its record at `render_record_path(r)` instead (the `record`
  field of [`render`](@ref)'s result).
"""
render_record_path(r::Render) =
    data_output_dir("renders", first(splitext(_render_name(r))) * ".jld2")

function render_record_path(image_path::AbstractString)
    image = normpath(abspath(image_path))
    base = normpath(abspath(output_dir())) * Base.Filesystem.path_separator
    inside =
        Sys.iswindows() ? startswith(lowercase(image), lowercase(base)) : startswith(image, base)
    rel = inside ? relpath(image, output_dir()) : basename(image)
    return data_output_dir("renders", first(splitext(rel)) * ".jld2")
end

# Write the PNG to `png` and the record (`r` and `limits`, git-tagged by DrWatson's `tagsave`) to
# `record`.
function _save_render(r::Render, img, limits, png::AbstractString, record::AbstractString)
    mkpath(dirname(png))
    _save_image_file(png, img)
    data = Dict{String,Any}("render" => r, "limits" => limits)
    if _SKIP_GIT[]  # precompile workload: no git subprocess
        DrWatson.wsave(record, data)
    else
        DrWatson.tagsave(record, data; gitpath = dirname(pkgdir(@__MODULE__)), warn = false)
    end
    return png, record
end

"""
    save_render(r::Render, img, limits; outpath = nothing) -> (outpath, record)

Write `img` as a PNG and the render record (`r` and `limits`, tagged with the Git commit by
DrWatson's `tagsave`) at [`render_record_path`](@ref)`(r)` in the data directory. `outpath`
places only the PNG; `outpath = nothing` names it from the IFS name, the method's parameters,
the resolution, and the color/alpha flags.
"""
save_render(r::Render, img, limits; outpath::Union{Nothing,AbstractString} = nothing) =
    _save_render(
        r,
        img,
        limits,
        _resolve_output_path(outpath, _render_name(r)),
        render_record_path(r),
    )

"""
    load_render(path) -> Render

The [`Render`](@ref) saved in a render record: a `.jld2` path, or a PNG whose record is at
[`render_record_path`](@ref)`(png)` (a default-located render or an animation frame). Run it
again with `render(load_render(path))`.
"""
function load_render(path::AbstractString)
    record = endswith(lowercase(path), ".jld2") ? path : render_record_path(path)
    return JLD2.load(record, "render")::Render
end

# Draw, finish, and save to explicit PNG and record paths.
function _render(r::Render, png::AbstractString, record::AbstractString)
    img, limits = draw(r)
    img = finish(r, img, limits)
    _save_render(r, img, limits, png, record)
    return (; render = r, image = img, limits, outpath = png, record)
end

# Whether two renders describe the same image. `==` on these structs falls back to identity for
# their vector fields, so compare the IFS by content key and the rest field by field (method
# structs and overlays hold only bits, symbols and strings, which `===` compares by value).
_same_render(a::Render, b::Render) =
    ifs_key(a.ifs) == ifs_key(b.ifs) &&
    a.ifs.limits == b.ifs.limits &&
    a.method === b.method &&
    a.resolution == b.resolution &&
    a.color == b.color &&
    a.alpha == b.alpha &&
    length(a.overlays) == length(b.overlays) &&
    all(o -> o[1] === o[2], zip(a.overlays, b.overlays))

# The saved image and limits of `r`, if its PNG and a matching record both exist.
function _load_existing_render(r::Render, png::AbstractString, record::AbstractString)
    (isfile(png) && isfile(record)) || return nothing
    saved = try
        JLD2.load(record)
    catch
        return nothing
    end
    stored = get(saved, "render", nothing)
    (stored isa Render && _same_render(stored, r)) || return nothing
    return _load_image(png), saved["limits"]
end

"""
    render(r::Render; outpath = nothing, force = false)
    render(ifs::IFS, method::RenderMethod; outpath = nothing, force = false, kwargs...)

Draw `r` with its method, apply color, overlays and alpha, then save the PNG and its render
record (see [`save_render`](@ref)). `outpath` places only the PNG; the record always goes to
[`render_record_path`](@ref)`(r)`. The second form builds the [`Render`](@ref) from `kwargs`.

Like DrWatson's `produce_or_load`: when the PNG and a record describing the same render
(same IFS content, method, resolution, color, alpha and overlays) already exist, they are
loaded instead of redrawn (`cached = true`; the image is then the PNG as read from disk).
`force = true` always redraws and overwrites both. Returns
`(render, image, limits, outpath, record, cached)`.
"""
function render(r::Render; outpath::Union{Nothing,AbstractString} = nothing, force::Bool = false)
    png = _resolve_output_path(outpath, _render_name(r))
    record = render_record_path(r)
    existing = force ? nothing : _load_existing_render(r, png, record)
    if !isnothing(existing)
        img, limits = existing
        return (; render = r, image = img, limits, outpath = png, record, cached = true)
    end
    return merge(_render(r, png, record), (; cached = false))
end

render(ifs::IFS, method::RenderMethod; outpath = nothing, force = false, kwargs...) =
    render(Render(ifs, method; kwargs...); outpath, force)
