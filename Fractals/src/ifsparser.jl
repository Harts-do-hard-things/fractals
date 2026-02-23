struct IFSToken
    kind::Symbol
    value
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
    inside = false
    for (lineno, raw) in enumerate(split(input, '\n'))
        line = rstrip(raw)
        isempty(line) && continue

        if occursin("(3D)", line)
            continue
        end

        if !inside
            if occursin('{', line)
                parts = split(line, '{', limit=2)
                name = replace(strip(parts[1]), "'" => "")
                if isempty(name)
                    throw(ArgumentError("Missing name before '{' at line $lineno"))
                end
                push!(tokens, IFSToken(:NAME, name, lineno))
                push!(tokens, IFSToken(:LBRACE, "{", lineno))
                inside = true

                if length(parts) > 1
                    rest = strip(parts[2])
                    if startswith(rest, ";")
                        push!(tokens, IFSToken(:DOCS, strip(rest[2:end]), lineno))
                    end
                end
            end
            continue
        end

        if occursin('}', line)
            push!(tokens, IFSToken(:RBRACE, "}", lineno))
            inside = false
            continue
        end

        if startswith(strip(line), ";")
            push!(tokens, IFSToken(:DOCS, strip(strip(line)[2:end]), lineno))
            continue
        end

        data = split(line, ';', limit=2)[1]
        data = strip(data)
        isempty(data) && continue
        nums = split(data)
        vals = Float64[]
        for n in nums
            push!(vals, parse(Float64, n))
        end
        push!(tokens, IFSToken(:ARRAY, vals, lineno))
    end

    if inside
        throw(EOFError("EOF while scanning IFS block: missing '}'"))
    end

    return tokens
end

function parse_ifs_tokens(tokens::Vector{IFSToken}; npoints::Integer=DEFAULT_SAMPLES)
    defs = IFS[]
    i = 1
    n = length(tokens)
    while i <= n
        tok = tokens[i]
        if tok.kind != :NAME
            throw(ArgumentError("Expected NAME at token $i (line $(tok.line))"))
        end
        name = tok.value
        i += 1

        if i > n || tokens[i].kind != :LBRACE
            throw(ArgumentError("Expected '{' after name '$name' (line $(tok.line))"))
        end
        i += 1

        docs_lines = String[]
        rows = Vector{Vector{Float64}}()
        while i <= n && tokens[i].kind != :RBRACE
            t = tokens[i]
            if t.kind == :DOCS
                push!(docs_lines, t.value)
            elseif t.kind == :ARRAY
                push!(rows, t.value)
            else
                throw(ArgumentError("Unexpected token $(t.kind) at line $(t.line)"))
            end
            i += 1
        end

        if i > n
            throw(EOFError("EOF while scanning IFS block for '$name': missing '}'"))
        end
        i += 1  # consume RBRACE

        if isempty(rows)
            throw(ArgumentError("IFS '$name' has no numeric rows"))
        end
        width = length(rows[1])
        for (idx, r) in enumerate(rows)
            if length(r) != width
                throw(ArgumentError("Row $idx for '$name' has length $(length(r)); expected $width"))
            end
        end
        eq = Matrix{Float64}(undef, length(rows), width)
        for (r, row) in enumerate(rows)
            eq[r, :] .= row
        end

        docs = join(docs_lines, "\n")
        push!(defs, IFS(eq; npoints=npoints, name=name, docs=docs))
    end
    return defs
end

function parse_ifs_string(input::AbstractString; npoints::Integer=DEFAULT_SAMPLES)
    return parse_ifs_tokens(lex_ifs(input); npoints=npoints)
end

function parse_ifs_file(path::AbstractString; npoints::Integer=DEFAULT_SAMPLES)
    return parse_ifs_string(read(path, String); npoints=npoints)
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
    println("  [1] chaos (iterate!)")
    println("  [2] chaos parallel (iterate_parallel!)")
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

    final_outpath = normalize_media_outpath(outpath)
    img = make_image(ifs; resolution=(height, width))
    save(final_outpath, img)
    println("Saved image to $(final_outpath)")
    return ifs
end
