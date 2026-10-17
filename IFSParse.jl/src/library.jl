# Source directory and the prebuilt JLD2 library (DrWatson-managed).

"""
    ifs_data_dir(args...)

Directory of the curated `.ifs` input files, `ifs/` at the DrWatson project root (the parent of
the `IFSParse.jl` package directory), joined with `args`. Unlike `projectdir("ifs")`, this does
not depend on which project is active, so tools run with `--project=Fractals.jl` find the same
files.
"""
ifs_data_dir(args::AbstractString...) = joinpath(dirname(pkgdir(@__MODULE__)), "ifs", args...)

# Default locations. Both follow the *active* DrWatson project (`projectdir()`), so under
# `--project=Fractals.jl` they resolve inside `Fractals.jl/`; pass `dir`/`libdir` to override.
_default_ifs_dir() = projectdir("ifs")
_default_library_dir() = datadir("exp_pro", "ifs")

_source_files(dir::AbstractString) =
    isdir(dir) ? sort!(filter(p -> endswith(lowercase(p), ".ifs"), readdir(dir; join = true))) :
    String[]

"""
    list_ifs_files(; dir = projectdir("ifs")) -> Vector{String}

Sorted paths of the `*.ifs` files in `dir` (empty if `dir` does not exist). `projectdir`
follows the active project; pass `dir` (e.g. [`ifs_data_dir`](@ref)`()`) to override.
"""
list_ifs_files(; dir::AbstractString = _default_ifs_dir()) = _source_files(dir)

"""
    build_ifs_library(; dir = projectdir("ifs"), outdir = datadir("exp_pro", "ifs"),
                      force = false, verbose = false) -> Vector{String}

Parse every `*.ifs` file in `dir` and save each function system as its own JLD2 record
`outdir/<source file stem>/ifs_<key>.jld2`, via DrWatson's `produce_or_load`. The record holds
`eq`, `weights`, `name`, `docs`, `source_file` and `key`.

The file name (and so the `produce_or_load` key) is `<key>`, a stable hash of the equation
matrix and weights only. The name is stored but is not part of the key: renaming a fractal in
its `.ifs` file does not create a new record, and two identical systems in one source file
share one record (the first name wins). Existing records are reused unless `force = true`.
Non-contractive definitions are skipped with a warning, as in [`parse_ifs_file`](@ref).
Returns the record paths, one per definition.

`projectdir` and `datadir` follow the active project, so under `--project=Fractals.jl` the
defaults point into `Fractals.jl/`; pass `dir` and `outdir` explicitly there.
"""
function build_ifs_library(;
    dir::AbstractString = _default_ifs_dir(),
    outdir::AbstractString = _default_library_dir(),
    force::Bool = false,
    verbose::Bool = false,
)
    paths = String[]
    for src in _source_files(dir)
        file = basename(src)
        stem = splitext(file)[1]
        defs = _validated_ifs_definitions(parse_ifs_definitions_file(src); source = src)
        for d in defs
            weights = _eq_weights(d.eq)
            key = _ifs_key(d.eq, weights)
            _, path = produce_or_load(
                Dict("key" => key),
                joinpath(outdir, stem);
                filename = key,
                prefix = "ifs",
                suffix = "jld2",
                tag = false,
                force = force,
                verbose = verbose,
                loadfile = false,
            ) do _
                Dict{String,Any}(
                    "eq" => copy(d.eq),
                    "weights" => weights,
                    "name" => d.name,
                    "docs" => d.docs,
                    "source_file" => file,
                    "key" => key,
                )
            end
            push!(paths, path)
        end
    end
    return paths
end

function _prebuilt_entries(libdir::AbstractString)
    entries = NamedTuple[]
    isdir(libdir) || return entries
    for sub in sort!(readdir(libdir; join = true))
        isdir(sub) || continue
        for path in sort!(filter(p -> endswith(p, ".jld2"), readdir(sub; join = true)))
            rec = try
                JLD2.load(path)
            catch
                continue
            end
            haskey(rec, "name") || continue
            push!(
                entries,
                (
                    name = String(rec["name"]),
                    file = String(get(rec, "source_file", basename(sub) * ".ifs")),
                    source = :prebuilt,
                    path = path,
                ),
            )
        end
    end
    return entries
end

function _ifs_file_entries(dir::AbstractString)
    entries = NamedTuple[]
    for src in _source_files(dir)
        valid, _ = _split_valid_ifs_definitions(parse_ifs_definitions_file(src))
        for d in valid
            push!(entries, (name = d.name, file = basename(src), source = :ifs_file, path = src))
        end
    end
    return entries
end

"""
    list_ifs(; dir = projectdir("ifs"), libdir = datadir("exp_pro", "ifs"),
             group_by_file = false)

List the available IFS definitions from every source: the prebuilt JLD2 records in `libdir`
(see [`build_ifs_library`](@ref)) and the `.ifs` files in `dir`. Returns a `Vector` of
`NamedTuple`s `(name, file, source, path)`, where `file` is the source `.ifs` file name,
`source` is `:prebuilt` or `:ifs_file`, and `path` is the JLD2 record or `.ifs` file. A
definition present in both (same `file` and `name`) is listed once, as `:prebuilt`.
Non-contractive definitions are left out silently.

With `group_by_file = true`, returns a `Vector` of `file => entries` pairs sorted by file name.
"""
function list_ifs(;
    dir::AbstractString = _default_ifs_dir(),
    libdir::AbstractString = _default_library_dir(),
    group_by_file::Bool = false,
)
    entries = _prebuilt_entries(libdir)
    seen = Set((e.file, e.name) for e in entries)
    for e in _ifs_file_entries(dir)
        (e.file, e.name) in seen || push!(entries, e)
    end
    sort!(entries; by = e -> (e.file, e.name, e.source))
    group_by_file || return entries
    files = unique(e.file for e in entries)
    return [f => filter(e -> e.file == f, entries) for f in files]
end

"""
    get_ifs_definition(name; file = nothing, dir = projectdir("ifs"),
                       libdir = datadir("exp_pro", "ifs")) -> IFSDefinition

Find the definition called `name` (optionally only in the source `.ifs` file `file`, e.g.
`"Default.ifs"`). The prebuilt JLD2 record is preferred; when none matches, the `.ifs` files in
`dir` are parsed. Throws `ArgumentError` if nothing matches. Because the name is not part of a
record's key, a record keeps the name it was first built with; run
[`build_ifs_library`](@ref)`(force = true)` to refresh it after renaming.
"""
function get_ifs_definition(
    name::AbstractString;
    file::Union{Nothing,AbstractString} = nothing,
    dir::AbstractString = _default_ifs_dir(),
    libdir::AbstractString = _default_library_dir(),
)
    matches(e) = e.name == name && (isnothing(file) || e.file == basename(file))
    for e in _prebuilt_entries(libdir)
        matches(e) || continue
        rec = JLD2.load(e.path)
        return IFSDefinition(rec["name"], rec["docs"], Matrix{Float64}(rec["eq"]))
    end
    for src in _source_files(dir)
        isnothing(file) || basename(src) == basename(file) || continue
        defs = _validated_ifs_definitions(parse_ifs_definitions_file(src); source = src)
        i = findfirst(d -> d.name == name, defs)
        i === nothing || return defs[i]
    end
    where_ = isnothing(file) ? "" : " in '$file'"
    throw(ArgumentError("No IFS named '$name'$where_ found in the prebuilt library or '$dir'"))
end
