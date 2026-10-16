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

"""
    parse_ifs_string(input; npoints=DEFAULT_SAMPLES) -> Vector{IFS}

Parse every block in `.ifs` text into an [`IFS`](@ref) with `npoints` points, keeping each
block's name and `;` docs. Non-contractive definitions are skipped with a warning. Syntax
errors throw as described in [`lex_ifs`](@ref); mixed row widths within a block and blocks
with no rows throw `ArgumentError`.

```jldoctest
julia> defs = parse_ifs_string(\"\"\"
       Heighway Dragon {
         0.5 -0.5  0.5  0.5  0.0 0.0
        -0.5 -0.5  0.5 -0.5  1.0 0.0
       }
       \"\"\"; npoints = 100);

julia> only(defs).name
"Heighway Dragon"
```
"""
function parse_ifs_string(input::AbstractString; npoints::Integer = DEFAULT_SAMPLES)
    defs = _validated_ifs_definitions(parse_ifs_definitions_string(input); source = "input")
    out = Vector{IFS}(undef, length(defs))
    @inbounds for i in eachindex(defs)
        d = defs[i]
        out[i] = IFS(d.eq; npoints = npoints, name = d.name, docs = d.docs)
    end
    return out
end

"""
    parse_ifs_file(path; npoints=DEFAULT_SAMPLES) -> Vector{IFS}

[`parse_ifs_string`](@ref) for the contents of the `.ifs` file at `path`; skip warnings
name the file.
"""
function parse_ifs_file(path::AbstractString; npoints::Integer = DEFAULT_SAMPLES)
    defs = _validated_ifs_definitions(parse_ifs_definitions_file(path); source = path)
    out = Vector{IFS}(undef, length(defs))
    @inbounds for i in eachindex(defs)
        d = defs[i]
        out[i] = IFS(d.eq; npoints = npoints, name = d.name, docs = d.docs)
    end
    return out
end
