"""
    IFSParse

The `.ifs` text format and its libraries: lexer and parser ([`lex_ifs`](@ref),
[`IFSDefinition`](@ref), [`parse_ifs_definitions_string`](@ref)), plus a
DrWatson-managed library of prebuilt definitions ([`build_ifs_library`](@ref),
[`list_ifs_files`](@ref), [`list_ifs`](@ref), [`get_ifs_definition`](@ref)).

A submodule of `Fractals` that depends on nothing else in it (definitions are plain equation
matrices); its public names are re-exported from there.
"""
module IFSParse

using DrWatson: projectdir, datadir, produce_or_load
import JLD2

export IFSToken,
    IFSDefinition,
    lex_ifs,
    parse_ifs_definitions_string,
    parse_ifs_definitions_file,
    ifs_data_dir,
    build_ifs_library,
    list_ifs_files,
    list_ifs,
    get_ifs_definition

"""
    IFSDefinition

One parsed `.ifs` block before it becomes an [`IFS`](@ref): its `name`, `docs` (the `;`
comment lines), and equation matrix `eq` (6 or 7 columns).
"""
struct IFSDefinition
    name::String
    docs::String
    eq::Matrix{Float64}
end

"""
    IFSToken

A lexer token from [`lex_ifs`](@ref): `kind` is one of `:NAME`, `:LBRACE`, `:DOCS`,
`:ARRAY` (one numeric row as `Vector{Float64}`), or `:RBRACE`; `value` holds the text or
numbers; `line` is the 1-based source line.
"""
struct IFSToken
    kind::Symbol
    value::Union{Nothing,String,Vector{Float64}}
    line::Int
end

"""
    lex_ifs(input::AbstractString) -> Vector{IFSToken}

Tokenize `.ifs` text (see the IFS text format page of the manual). Lines containing `(3D)` are
skipped. Throws `ArgumentError` (with the line number) for a `}` outside a block, a `{`
without a name, or a non-numeric token in a row, and `EOFError` for an unclosed block.
"""
function lex_ifs(input::AbstractString)
    tokens = IFSToken[]
    sizehint!(tokens, max(16, div(length(input), 32)))
    inside = false
    for (lineno, raw) in enumerate(eachline(IOBuffer(input)))
        line = rstrip(raw)
        isempty(line) && continue

        if occursin("(3D)", line)
            continue
        end

        if !inside
            if occursin('}', line)
                throw(ArgumentError("Unexpected '}' outside IFS block at line $lineno"))
            end
            if occursin('{', line)
                parts = split(line, '{', limit = 2)
                name = replace(strip(parts[1]), "'" => "")
                if isempty(name)
                    throw(ArgumentError("Missing name before '{' at line $lineno"))
                end
                push!(tokens, IFSToken(:NAME, name, lineno))
                push!(tokens, IFSToken(:LBRACE, nothing, lineno))
                inside = true

                if length(parts) > 1
                    rest = strip(parts[2])
                    if startswith(rest, ";")
                        push!(tokens, IFSToken(:DOCS, String(strip(rest[2:end])), lineno))
                    end
                end
            end
            continue
        end

        if occursin('}', line)
            push!(tokens, IFSToken(:RBRACE, nothing, lineno))
            inside = false
            continue
        end

        if startswith(strip(line), ";")
            push!(tokens, IFSToken(:DOCS, String(strip(strip(line)[2:end])), lineno))
            continue
        end

        data = split(line, ';', limit = 2)[1]
        data = strip(data)
        isempty(data) && continue
        nums = split(data)
        vals = Float64[]
        for n in nums
            try
                push!(vals, parse(Float64, n))
            catch
                throw(ArgumentError("Invalid numeric token '$n' at line $lineno"))
            end
        end
        push!(tokens, IFSToken(:ARRAY, vals, lineno))
    end

    if inside
        throw(EOFError())
    end

    return tokens
end

function _parse_ifs_tokens(tokens::Vector{IFSToken})
    defs = IFSDefinition[]
    i = 1
    n = length(tokens)
    while i <= n
        tok = tokens[i]
        if tok.kind != :NAME
            throw(ArgumentError("Expected NAME at token $i (line $(tok.line))"))
        end
        name = tok.value::String
        i += 1

        if i > n || tokens[i].kind != :LBRACE
            throw(ArgumentError("Expected '{' after name '$name' (line $(tok.line))"))
        end
        i += 1

        docs_lines = String[]
        flat_vals = Float64[]
        width = 0
        nrows = 0
        while i <= n && tokens[i].kind != :RBRACE
            t = tokens[i]
            if t.kind == :DOCS
                push!(docs_lines, t.value::String)
            elseif t.kind == :ARRAY
                row = t.value::Vector{Float64}
                row_width = length(row)
                if width == 0
                    width = row_width
                elseif row_width != width
                    throw(
                        ArgumentError(
                            "Row $(nrows + 1) for '$name' has length $row_width; expected $width",
                        ),
                    )
                end
                append!(flat_vals, row)
                nrows += 1
            else
                throw(ArgumentError("Unexpected token $(t.kind) at line $(t.line)"))
            end
            i += 1
        end

        if i > n
            throw(EOFError())
        end
        i += 1  # consume RBRACE

        if nrows == 0
            throw(ArgumentError("IFS '$name' has no numeric rows"))
        end
        eq = Matrix{Float64}(undef, nrows, width)
        k = 1
        @inbounds for r = 1:nrows
            for c = 1:width
                eq[r, c] = flat_vals[k]
                k += 1
            end
        end

        docs = join(docs_lines, "\n")
        push!(defs, IFSDefinition(name, docs, eq))
    end
    return defs
end

function parse_ifs_definitions_string(input::AbstractString)
    return _parse_ifs_tokens(lex_ifs(input))
end

function parse_ifs_definitions_file(path::AbstractString)
    return parse_ifs_definitions_string(read(path, String))
end

function _split_valid_ifs_definitions(defs::Vector{IFSDefinition})
    valid = IFSDefinition[]
    invalid = NamedTuple[]
    for d in defs
        failures = _noncontractive_rows(d.eq)
        if isempty(failures)
            push!(valid, d)
        else
            push!(invalid, (definition = d, failures = failures))
        end
    end
    return valid, invalid
end

function _warn_skipped_invalid_ifs(source::AbstractString, invalid)
    for entry in invalid
        @warn "Skipping non-contractive IFS definition '$(entry.definition.name)' from $source: $(_format_contractivity_failure(first(entry.failures)))."
    end
    return nothing
end

function _validated_ifs_definitions(defs::Vector{IFSDefinition}; source::AbstractString = "input")
    valid, invalid = _split_valid_ifs_definitions(defs)
    isempty(invalid) || _warn_skipped_invalid_ifs(source, invalid)
    return valid
end

# ---------------------------------------------------------------------------------------
# Source directory and prebuilt JLD2 library
# ---------------------------------------------------------------------------------------

"""
    ifs_data_dir(args...)

Directory of the curated `.ifs` input files, `ifs/` at the DrWatson project root (the parent of
the `Fractals.jl` package directory), joined with `args`. Unlike `projectdir("ifs")`, this does
not depend on which project is active, so tools run with `--project=Fractals.jl` find the same
files.
"""
ifs_data_dir(args::AbstractString...) = joinpath(dirname(pkgdir(@__MODULE__)), "ifs", args...)

# Default locations. Both follow the *active* DrWatson project (`projectdir()`), so under
# `--project=Fractals.jl` they resolve inside `Fractals.jl/`; pass `dir`/`libdir` to override.
_default_ifs_dir() = projectdir("ifs")
_default_library_dir() = datadir("exp_pro", "ifs")

# 64-bit FNV-1a over the bit patterns of the numbers: stable across sessions and Julia
# versions, unlike `Base.hash`.
function _fnv_step(h::UInt64, x::UInt64)
    for shift = 0:8:56
        h = xor(h, (x >> shift) & 0xff) * 0x00000100000001b3
    end
    return h
end

# `+ 0.0` maps -0.0 to 0.0 so that equal systems hash equally.
_bits(v::Real) = reinterpret(UInt64, Float64(v) + 0.0)

# Content key of a function system: a 16-hex-digit hash of the 6 map columns of `eq` and the
# sampling `weights`. The name, docs and source file are deliberately not part of it.
function _ifs_key(eq::AbstractMatrix{<:Real}, weights::AbstractVector{<:Real})
    h = 0xcbf29ce484222325
    h = _fnv_step(h, UInt64(size(eq, 1)))
    for v in view(eq, :, 1:6)
        h = _fnv_step(h, _bits(v))
    end
    for w in weights
        h = _fnv_step(h, _bits(w))
    end
    return string(h; base = 16, pad = 16)
end

# Sampling weights of an equation matrix: the 7th column if present, else |det(A)| per map.
function _eq_weights(eq::AbstractMatrix{<:Real})
    size(eq, 2) == 7 && return Float64.(eq[:, 7])
    return [abs(eq[i, 1] * eq[i, 4] - eq[i, 2] * eq[i, 3]) for i = 1:size(eq, 1)] .+ 0.0
end

# Contractivity of the maps (rows `a11 a12 a21 a22 ...`; the check uses `a, b, d, e` =
# columns 1-4), used to skip unusable definitions.
@inline function _contractive_row(a::Real, b::Real, d::Real, e::Real)
    aa = Float64(a)
    bb = Float64(b)
    dd = Float64(d)
    ee = Float64(e)
    return aa^2 + dd^2 < 1 &&
           bb^2 + ee^2 < 1 &&
           aa^2 + bb^2 + dd^2 + ee^2 < 1 + (aa * ee - dd * bb)^2
end

function _noncontractive_rows(eq::AbstractMatrix{<:Real})
    failures = NamedTuple[]
    for i = 1:size(eq, 1)
        a, b, d, e = eq[i, 1], eq[i, 2], eq[i, 3], eq[i, 4]
        _contractive_row(a, b, d, e) && continue
        push!(failures, (row = i, a = Float64(a), b = Float64(b), d = Float64(d), e = Float64(e)))
    end
    return failures
end

function _format_contractivity_failure(failure)
    return "row $(failure.row) is not contractive for coefficients (a=$(failure.a), b=$(failure.b), d=$(failure.d), e=$(failure.e))"
end

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

end # module IFSParse
