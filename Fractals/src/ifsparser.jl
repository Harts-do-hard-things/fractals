struct IFSDefinition
    name::String
    docs::String
    eq::Matrix{Float64}
end

struct IFSToken
    kind::Symbol
    value::Union{Nothing,String,Vector{Float64}}
    line::Int
end

function _prompt_int(msg::AbstractString, default::Int)
    print(msg)
    line = readline()
    isempty(strip(line)) && return default
    try
        return parse(Int, strip(line))
    catch
        println("Invalid number, using default $default")
        return default
    end
end

function _prompt_string(msg::AbstractString, default::AbstractString)
    print(msg)
    line = readline()
    isempty(strip(line)) && return String(default)
    return strip(line)
end

function _prompt_choice(msg::AbstractString, n::Int, default::Int)
    print(msg)
    line = readline()
    isempty(strip(line)) && return default
    try
        choice = parse(Int, strip(line))
        if 1 <= choice <= n
            return choice
        end
    catch
    end
    println("Invalid choice, using default $default")
    return default
end

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
                parts = split(line, '{', limit=2)
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

        data = split(line, ';', limit=2)[1]
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
                    throw(ArgumentError("Row $(nrows + 1) for '$name' has length $row_width; expected $width"))
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
        @inbounds for r in 1:nrows
            for c in 1:width
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

function parse_ifs_string(input::AbstractString; npoints::Integer=DEFAULT_SAMPLES)
    defs = parse_ifs_definitions_string(input)
    out = Vector{IFS}(undef, length(defs))
    @inbounds for i in eachindex(defs)
        d = defs[i]
        out[i] = IFS(d.eq; npoints=npoints, name=d.name, docs=d.docs)
    end
    return out
end

function parse_ifs_file(path::AbstractString; npoints::Integer=DEFAULT_SAMPLES)
    defs = parse_ifs_definitions_file(path)
    out = Vector{IFS}(undef, length(defs))
    @inbounds for i in eachindex(defs)
        d = defs[i]
        out[i] = IFS(d.eq; npoints=npoints, name=d.name, docs=d.docs)
    end
    return out
end

function prompt_ifs_and_render(path::AbstractString;
                               npoints::Integer=DEFAULT_SAMPLES,
                               resolution::Tuple{Int,Int}=RESOLUTION,
                               outpath::AbstractString="media/output.png")
    ifs_list = parse_ifs_file(path; npoints=npoints)
    if isempty(ifs_list)
        println("No IFS definitions found in: $path")
        return nothing
    end

    println("IFS definitions in file:")
    for (i, ifs) in enumerate(ifs_list)
        doc_preview = isempty(ifs.docs) ? "" : " - " * first(split(ifs.docs, '\n'))
        println("  [$i] $(ifs.name)$(doc_preview)")
    end

    idx = _prompt_choice("Select a fractal [1-$(length(ifs_list))] (default 1): ",
                         length(ifs_list), 1)
    ifs = ifs_list[idx]

    println("Iteration methods:")
    println("  [1] chaos auto-threaded (iterate!)")
    println("  [2] chaos auto-threaded (iterate_parallel! alias)")
    println("  [3] deterministic (deterministic_iterate, n=1)")
    method = _prompt_choice("Select method [1-3] (default 1): ", 3, 1)

    npoints = _prompt_int("Number of points (default $(npoints)): ", npoints)
    width = _prompt_int("Image width (default $(resolution[2])): ", resolution[2])
    height = _prompt_int("Image height (default $(resolution[1])): ", resolution[1])
    outpath = _prompt_string("Output path (default $(outpath)): ", outpath)

    ifs = IFS(ifs.maps, ifs.weights; npoints=npoints, name=ifs.name, docs=ifs.docs, limits=ifs.limits)

    if method == 1
        iterate!(ifs)
    elseif method == 2
        iterate_parallel!(ifs)
    else
        ifs = deterministic_iterate(ifs, 1)
    end

    final_outpath = _normalize_media_outpath(outpath)
    img = make_image(ifs; resolution=(height, width))
    save(final_outpath, img)
    println("Saved image to $(final_outpath)")
    return ifs
end
