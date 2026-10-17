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
