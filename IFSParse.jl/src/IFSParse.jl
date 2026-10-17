"""
    IFSParse

The `.ifs` text format and its libraries: lexer and parser ([`lex_ifs`](@ref),
[`IFSDefinition`](@ref), [`parse_ifs_definitions_string`](@ref)), plus a
DrWatson-managed library of prebuilt definitions ([`build_ifs_library`](@ref),
[`list_ifs_files`](@ref), [`list_ifs`](@ref), [`get_ifs_definition`](@ref)).

A standalone package: definitions are plain equation matrices, so it does not depend on
`Fractals`, which re-exports its public names.
"""
module IFSParse

using DrWatson: projectdir, datadir, produce_or_load
import JLD2

include("types.jl")    # IFSDefinition, IFSToken
include("parse.jl")    # lexer, parser, contractivity, weights, content key
include("library.jl")  # ifs_data_dir, build_ifs_library, list_ifs*, get_ifs_definition

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

end # module IFSParse
