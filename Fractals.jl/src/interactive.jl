# NOTE: This file must be included after ifsparser.jl (calls parse_ifs_file) and after all
# render-method files (prompt_ifs_and_render calls iterate!, deterministic_iterate, make_image,
# _normalize_media_outpath). It must be included BEFORE render.jl — render.jl's
# _resolve_render_input(AbstractString) calls _select_ifs_definition defined here.

# -------------------------------------------------
# Prompt primitives (injectable stdin)
# -------------------------------------------------

function _prompt_int(msg::AbstractString, default::Int; input_fn=readline)
    print(msg)
    line = input_fn()
    isempty(strip(line)) && return default
    try
        return parse(Int, strip(line))
    catch
        println("Invalid number, using default $default")
        return default
    end
end

function _prompt_string(msg::AbstractString, default::AbstractString; input_fn=readline)
    print(msg)
    line = input_fn()
    isempty(strip(line)) && return String(default)
    return strip(line)
end

function _prompt_choice(msg::AbstractString, n::Int, default::Int; input_fn=readline)
    print(msg)
    line = input_fn()
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

# -------------------------------------------------
# IFS definition selection (used by render.jl)
# -------------------------------------------------

function _ifs_choices_text(defs::AbstractVector)
    lines = ["Available IFS definitions:"]
    for (i, d) in enumerate(defs)
        push!(lines, "  [$i] $(d.name)")
    end
    return join(lines, "\n")
end

function _select_ifs_definition(
    defs::AbstractVector;
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString},
    input_fn=readline
)
    choices = _ifs_choices_text(defs)

    if !isnothing(ifs_index) && !isnothing(ifs_name)
        throw(ArgumentError("Provide only one of ifs_index or ifs_name.\n$choices"))
    end

    if !isnothing(ifs_name)
        for d in defs
            if d.name == ifs_name
                return d
            end
        end
        throw(ArgumentError("IFS name '$ifs_name' not found.\n$choices"))
    end

    if !isnothing(ifs_index)
        if ifs_index <= length(defs)
            return defs[ifs_index]
        end
        throw(ArgumentError("ifs_index=$ifs_index is out of range (1-$(length(defs))).\n$choices"))
    end

    if length(defs) == 1
        return defs[1]
    end

    println(choices)
    print("No ifs_index/ifs_name provided. Render [1] $(defs[1].name)? [y/N]: ")
    answer = try
        lowercase(strip(input_fn()))
    catch
        ""
    end

    if answer in ("y", "yes")
        return defs[1]
    end

    throw(ArgumentError("No IFS selection confirmed.\n$choices"))
end

# -------------------------------------------------
# Interactive IFS selection + render entrypoint
# -------------------------------------------------

function prompt_ifs_and_render(path::AbstractString;
                               npoints::Integer=DEFAULT_SAMPLES,
                               resolution::Tuple{Int,Int}=RESOLUTION,
                               outpath::AbstractString="media/output.png",
                               input_fn=readline)
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
                         length(ifs_list), 1; input_fn=input_fn)
    ifs = ifs_list[idx]

    println("Iteration methods:")
    println("  [1] chaos auto-threaded (iterate!)")
    println("  [2] chaos auto-threaded (iterate_parallel! alias)")
    println("  [3] point deterministic (deterministic_iterate, n=1)")
    method = _prompt_choice("Select method [1-3] (default 1): ", 3, 1; input_fn=input_fn)

    npoints = _prompt_int("Number of points (default $(npoints)): ", npoints; input_fn=input_fn)
    width   = _prompt_int("Image width (default $(resolution[2])): ", resolution[2]; input_fn=input_fn)
    height  = _prompt_int("Image height (default $(resolution[1])): ", resolution[1]; input_fn=input_fn)
    outpath = _prompt_string("Output path (default $(outpath)): ", outpath; input_fn=input_fn)

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
    _save_image_with_source_limits(final_outpath, img, ifs.limits)
    println("Saved image to $(final_outpath)")
    return ifs
end
