# Lexer, parser, validation, weights and the content key hash (no I/O besides reading files).

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

"""
    parse_ifs_definitions_string(input) -> Vector{IFSDefinition}

Lex and parse `.ifs` text into raw [`IFSDefinition`](@ref)s (name, docs, equation matrix). Unlike
`parse_ifs_string` this builds no `IFS` and does not drop non-contractive definitions. Errors
are those of [`lex_ifs`](@ref) and the block parser.
"""
function parse_ifs_definitions_string(input::AbstractString)
    return _parse_ifs_tokens(lex_ifs(input))
end

"""
    parse_ifs_definitions_file(path) -> Vector{IFSDefinition}

[`parse_ifs_definitions_string`](@ref) for the contents of the file at `path`.
"""
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
