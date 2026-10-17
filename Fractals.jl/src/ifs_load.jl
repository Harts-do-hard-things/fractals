# IFS-returning wrappers around the IFSParse package (which only knows plain equation
# matrices). Must follow types.jl (IFS) and the IFSParse package.

"""
    parse_ifs_string(input) -> Vector{IFS}

Parse every block in `.ifs` text into an [`IFS`](@ref), keeping each
block's name. Non-contractive definitions are skipped with a warning. Syntax
errors throw as described in [`lex_ifs`](@ref); mixed row widths within a block and blocks
with no rows throw `ArgumentError`.

```jldoctest
julia> defs = parse_ifs_string(\"\"\"
       Heighway Dragon {
         0.5 -0.5  0.5  0.5  0.0 0.0
        -0.5 -0.5  0.5 -0.5  1.0 0.0
       }
       \"\"\");

julia> only(defs).name
"Heighway Dragon"
```
"""
function parse_ifs_string(input::AbstractString)
    defs = _validated_ifs_definitions(parse_ifs_definitions_string(input); source = "input")
    out = Vector{IFS}(undef, length(defs))
    @inbounds for i in eachindex(defs)
        d = defs[i]
        out[i] = IFS(d.eq; name = d.name)
    end
    return out
end

"""
    parse_ifs_file(path) -> Vector{IFS}

[`parse_ifs_string`](@ref) for the contents of the `.ifs` file at `path`; skip warnings
name the file.
"""
function parse_ifs_file(path::AbstractString)
    defs = _validated_ifs_definitions(parse_ifs_definitions_file(path); source = path)
    out = Vector{IFS}(undef, length(defs))
    @inbounds for i in eachindex(defs)
        d = defs[i]
        out[i] = IFS(d.eq; name = d.name)
    end
    return out
end


"""
    get_ifs(name; file = nothing, dir = projectdir("ifs"), libdir = datadir("exp_pro", "ifs")) -> IFS

Find the definition called `name` (optionally only in the source `.ifs` file `file`, e.g.
`"Default.ifs"`) and build an [`IFS`](@ref). The prebuilt JLD2 record
(see [`build_ifs_library`](@ref)) is preferred; when none matches, the `.ifs` files in `dir` are
parsed. Throws `ArgumentError` if nothing matches. A record keeps the name it was first built
with (the name is not part of its key); run `build_ifs_library(force = true)` to refresh it.
"""
function get_ifs(
    name::AbstractString;
    file::Union{Nothing,AbstractString} = nothing,
    dir::AbstractString = IFSParse._default_ifs_dir(),
    libdir::AbstractString = IFSParse._default_library_dir(),
)
    d = get_ifs_definition(name; file = file, dir = dir, libdir = libdir)
    return IFS(d.eq; name = d.name)
end
