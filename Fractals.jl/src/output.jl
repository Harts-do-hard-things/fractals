# --------------------------------
# Output management (DrWatson)
# --------------------------------
#
# Every file the package writes goes through this file:
# - Relative and auto-generated paths resolve under DrWatson's `plotsdir()` for the active
#   project (override with `with_output_root` or `FRACTALS_OUTPUT_ROOT`); absolute paths are kept.
# - Omitted output paths are named from render parameters with `savename`.
# - Existing files are never overwritten: `safesave` moves them to `name_#N.ext` backups.
# - PNGs carry source-limits and provenance (git commit + parameters) text chunks; SVGs carry
#   the provenance as a leading XML comment.

const _OUTPUT_ROOT = ScopedValue{Union{Nothing,String}}(nothing)
const _PNG_PROVENANCE_KEYWORD = "fractals.provenance"
const _PROVENANCE_VERSION = 1

# Parameters recorded in provenance but left out of generated filenames because they do not
# change the rendered content.
const _NAME_EXCLUDED_KEYS = (:backend,)

"""
    with_output_root(f, root)

Run `f()` with package output resolved under `root` (`root/plots`, `root/data/exp_pro`) instead
of the active DrWatson project's directories. Scoped to `f` (and tasks it spawns).
"""
with_output_root(f, root::AbstractString) = with(f, _OUTPUT_ROOT => abspath(root))

# Override root from `with_output_root` or `FRACTALS_OUTPUT_ROOT`, or `nothing` to use the
# active DrWatson project.
function _output_root_override()
    root = _OUTPUT_ROOT[]
    isnothing(root) || return root
    env_root = strip(get(ENV, "FRACTALS_OUTPUT_ROOT", ""))
    return isempty(env_root) ? nothing : abspath(env_root)
end

"""
    output_dir(args...)

Directory that relative and auto-generated image/animation paths resolve against, joined with
`args`: DrWatson's `plotsdir()` for the active project, or `<root>/plots` when overridden by
`with_output_root` or the `FRACTALS_OUTPUT_ROOT` environment variable.
"""
function output_dir(args::AbstractString...)
    root = _output_root_override()
    return isnothing(root) ? plotsdir(args...) : joinpath(root, "plots", args...)
end

"""
    data_output_dir(args...)

Directory for generated (processed) data such as benchmark JSON, joined with `args`: DrWatson's
`datadir("exp_pro")` for the active project, or `<root>/data/exp_pro` when overridden like
[`output_dir`](@ref).
"""
function data_output_dir(args::AbstractString...)
    root = _output_root_override()
    return isnothing(root) ? datadir("exp_pro", args...) :
           joinpath(root, "data", "exp_pro", args...)
end

"""
    ifs_data_dir(args...)

Directory of the curated `.ifs` input library, `data/exp_raw` at the DrWatson project root (the
parent of the `Fractals.jl` package directory), joined with `args`. Unlike `datadir()`, this
does not depend on which project is active, so tests and tools run with
`--project=Fractals.jl` find the same files.
"""
ifs_data_dir(args::AbstractString...) =
    joinpath(dirname(pkgdir(@__MODULE__)), "data", "exp_raw", args...)

"""
    _resolve_output_path(outpath, default_name; base=output_dir, prefix="plots")

`nothing` → `base(default_name)`; absolute paths are returned unchanged; relative paths resolve
under `base()` (a leading `prefix` component is accepted and not duplicated).
"""
function _resolve_output_path(
    outpath::Union{Nothing,AbstractString},
    default_name::AbstractString;
    base = output_dir,
    prefix::AbstractString = "plots",
)
    isnothing(outpath) && return base(default_name)
    path = normpath(String(outpath))
    isabspath(path) && return path
    parts = splitpath(path)
    if !isempty(parts) && first(parts) == prefix
        parts = parts[2:end]
    end
    (isempty(parts) || parts == ["."]) &&
        throw(ArgumentError("Output path '$outpath' does not name a file or directory"))
    return base(parts...)
end

"""
    _save_data_text(outpath, text)

Resolve `outpath` under `data_output_dir()` (relative paths) and `safesave` `text` there.
Returns the final path.
"""
function _save_data_text(outpath::AbstractString, text::AbstractString)
    final_outpath = _resolve_output_path(outpath, ""; base = data_output_dir, prefix = "data")
    safesave(final_outpath, _TextOutput(String(text)))
    return final_outpath
end

"""
    _resolve_data_path(outpath, default_name)

Like `_resolve_output_path` for generated data: `nothing` and relative paths resolve under
`data_output_dir()` (a leading `data/` is not duplicated).
"""
_resolve_data_path(outpath::Union{Nothing,AbstractString}, default_name::AbstractString) =
    _resolve_output_path(outpath, default_name; base = data_output_dir, prefix = "data")

# DrWatson-style tagging (`tagsave` adds a "gitcommit" key), done here with the cached
# `_gitcommit` because `tagsave` rescans the work tree on every save.
function _tag_record!(record::AbstractDict)
    commit = _gitcommit()
    isnothing(commit) || haskey(record, "gitcommit") || (record["gitcommit"] = commit)
    return record
end

"""
    _save_data_record(path, record::AbstractDict; safe=true)

Save a JLD2 `record` at the already-resolved `path` through DrWatson, tagged with the Git
commit (`"gitcommit"`, as DrWatson's `tagsave` does) when the project is in a Git repository.
With `safe=true` (`safesave`), an existing file is kept as a `_#N` backup. Returns the path.
"""
function _save_data_record(path::AbstractString, record::AbstractDict; safe::Bool = true)
    tagged = _tag_record!(copy(record))
    safe ? safesave(path, tagged) : DrWatson.wsave(path, tagged)
    return path
end

"""
    _load_data_record(path)

Load a JLD2 record saved by `_save_data_record` through DrWatson's `wload`.
"""
_load_data_record(path::AbstractString) = DrWatson.wload(path)

"""
    _produce_or_load_record(produce, config, filename; force=false)

DrWatson `produce_or_load` into `data_output_dir()`: if `filename` (a `.jld2` name) exists it
is loaded, otherwise `produce(config)` must return a record `Dict`, which is saved (tagged
with the Git commit when in a repository). Returns `(record, path, cached)`.
"""
function _produce_or_load_record(
    produce::Function,
    config,
    filename::AbstractString;
    force::Bool = false,
)
    produced = Ref(false)
    record, path = DrWatson.produce_or_load(
        c -> (produced[] = true; _tag_record!(produce(c))),
        config,
        data_output_dir();
        filename = String(filename),
        prefix = "",
        suffix = "",
        tag = false,
        force = force,
        verbose = false,
    )
    return record, path, !produced[]
end

function _savename_prefix(name::AbstractString)
    prefix = strip(replace(strip(String(name)), r"[^A-Za-z0-9.\-]+" => "_"), '_')
    return isempty(prefix) ? "ifs" : String(prefix)
end

_savename_value(v::Tuple{Integer,Integer}) = "$(v[1])x$(v[2])"
_savename_value(v) = v

"""
    _output_name(name, params, ext)

DrWatson `savename` for an output file: `name` (sanitized) as prefix, `params` as `key=value`
pairs. `false`/`nothing` values and `_NAME_EXCLUDED_KEYS` are dropped to keep names short.
"""
Base.@nospecializeinfer function _output_name(
    name::AbstractString,
    @nospecialize(params::NamedTuple),
    ext::AbstractString,
)
    kept = Dict{String,Any}()
    for (k, v) in pairs(params)
        (k in _NAME_EXCLUDED_KEYS || v === nothing || v === false) && continue
        kept[String(k)] = _savename_value(v)
    end
    return savename(_savename_prefix(name), kept, ext)
end

function _in_git_repo(dir::AbstractString)
    current = abspath(dir)
    while true
        ispath(joinpath(current, ".git")) && return true
        parent = dirname(current)
        parent == current && return false
        current = parent
    end
end

# `gitdescribe` scans the working tree for the dirty flag (~60 ms), so results are reused briefly
# across back-to-back saves (batch renders, GUI preview refreshes).
# The precompile workload sets this: shelling out to `git` there is slow and records nothing useful.
const _SKIP_GIT = ScopedValue(false)
const _GITCOMMIT_TTL_SECONDS = 2.0
const _GITCOMMIT_CACHE = Ref{Tuple{String,Float64,Union{Nothing,String}}}(("", -Inf, nothing))

"""
    _gitcommit()

`gitdescribe` of the active project (e.g. `v1.0.1-40-gaec1219-dirty`), or `nothing` when the
project is not inside a Git repository.
"""
function _gitcommit()
    _SKIP_GIT[] && return nothing
    root = projectdir()
    cached_root, stamp, cached = _GITCOMMIT_CACHE[]
    cached_root == root && time() - stamp < _GITCOMMIT_TTL_SECONDS && return cached
    value = _in_git_repo(root) ? gitdescribe(root; warn = false) : nothing
    _GITCOMMIT_CACHE[] = (root, time(), value)
    return value
end

_provenance_value(v::Symbol) = String(v)
_provenance_value(v::Tuple) = collect(v)
_provenance_value(v) = v

Base.@nospecializeinfer function _provenance_payload(
    @nospecialize(params::NamedTuple);
    gitcommit = _gitcommit(),
)
    # A fixed field type for `gitcommit` (String or nothing) keeps one JSON3.write specialization.
    payload = (;
        version = _PROVENANCE_VERSION,
        gitcommit = gitcommit isa Union{Nothing,String} ? gitcommit : string(gitcommit),
        created = string(now()),
        fractals_version = string(pkgversion(@__MODULE__)),
        params = Dict{String,Any}(String(k) => _provenance_value(v) for (k, v) in pairs(params)),
    )
    return JSON3.write(
        NamedTuple{keys(payload),Tuple{Int,Union{Nothing,String},String,String,Dict{String,Any}}}(
            values(payload),
        ),
    )
end

"""
    read_provenance(path)

Parsed provenance metadata embedded in a PNG written by this package, or `nothing`.
"""
function read_provenance(path::AbstractString)
    payload = _read_png_text_chunk(path, _PNG_PROVENANCE_KEYWORD)
    isnothing(payload) && return nothing
    return JSON3.read(payload)
end

# DrWatson `_wsave` payloads: `safesave` handles backups and parent directories, these write.
struct _PNGOutput{I}
    image::I
    text_chunks::Vector{Pair{String,String}}
end

struct _TextOutput
    text::String
end

struct _MovedFileOutput
    source::String
end

function DrWatson._wsave(path, out::_PNGOutput)
    _save_image_file(path, out.image)
    _has_png_extension(path) && _embed_png_text_chunks(path, out.text_chunks)
    return path
end

DrWatson._wsave(path, out::_TextOutput) = write(path, out.text)
DrWatson._wsave(path, out::_MovedFileOutput) = mv(out.source, path)

"""
    _save_image(outpath, img, limits; params, provenance)

`safesave` an image to the already-resolved `outpath`, embedding source limits and provenance
when it is a PNG. Returns `outpath`.
"""
function _save_image(
    outpath::AbstractString,
    img,
    limits;
    params::NamedTuple = (;),
    provenance::AbstractString = _provenance_payload(params),
)
    chunks = [
        _PNG_SOURCE_LIMITS_KEYWORD => _serialize_source_limits_payload(limits),
        _PNG_PROVENANCE_KEYWORD => String(provenance),
    ]
    safesave(outpath, _PNGOutput(img, chunks))
    return outpath
end

"""
    _save_svg(outpath, svg; params)

`safesave` SVG text to the already-resolved `outpath`, prefixed with a provenance comment.
"""
function _save_svg(outpath::AbstractString, svg::AbstractString; params::NamedTuple = (;))
    provenance = replace(_provenance_payload(params), "--" => "- -")
    safesave(outpath, _TextOutput("<!-- $(_PNG_PROVENANCE_KEYWORD): $provenance -->\n" * svg))
    return outpath
end

"""
    _safesave_moved_file(outpath, source)

Move a finished file (e.g. ffmpeg output) into `outpath`, backing up any existing file.
"""
function _safesave_moved_file(outpath::AbstractString, source::AbstractString)
    safesave(outpath, _MovedFileOutput(String(source)))
    return outpath
end
