module FractalsGUI

using Fractals
using LinearAlgebra
using Printf
import Gtk

export GUIState,
       launch_gui,
       make_default_state,
       normalize_eq_matrix,
       list_data_ifs,
       list_ifs_definition_names,
       parse_matrix_text,
       matrix_to_text,
       parse_ifs_definition_for_gui,
       load_ifs_definition,
       load_ifs_definition!,
       load_ifs_definition_gui!,
       apply_eq_matrix!

const _DEFAULT_ROW = [1.0 0.0 0.0 1.0 0.0 0.0 1.0]

mutable struct GUIState
    source_file::Union{Nothing,String}
    definition_index::Int
    definition_name::String
    eq_matrix::Matrix{Float64}
    loaded_eq_matrix::Matrix{Float64}
    docs::String
    ifs::IFS
    svg_string::String
    svg_temp_path::String
    fractal_placeholder_text::String
    last_error::Union{Nothing,String}
    is_valid::Bool
end

function normalize_eq_matrix(eq::AbstractMatrix{<:Real})::Matrix{Float64}
    nrows, ncols = size(eq)
    nrows > 0 || throw(ArgumentError("matrix must have at least one row"))
    (ncols == 6 || ncols == 7) || throw(ArgumentError("matrix must have 6 or 7 columns, got $ncols"))

    out = Matrix{Float64}(undef, nrows, 7)
    out[:, 1:6] .= Float64.(eq[:, 1:6])
    all(isfinite, out[:, 1:6]) || throw(ArgumentError("matrix contains non-finite values"))

    if ncols == 7
        out[:, 7] .= Float64.(eq[:, 7])
    else
        for i in 1:nrows
            a11, a12, a21, a22 = out[i, 1], out[i, 2], out[i, 3], out[i, 4]
            out[i, 7] = abs(det([a11 a12; a21 a22]))
        end
    end

    all(isfinite, out[:, 7]) || throw(ArgumentError("probability column contains non-finite values"))
    all(p -> p >= 0.0, out[:, 7]) || throw(ArgumentError("probability column must be nonnegative"))
    sum(out[:, 7]) > 0.0 || throw(ArgumentError("probability column must have positive total weight"))
    return out
end

function list_data_ifs(data_dir::AbstractString)::Vector{String}
    isdir(data_dir) || return String[]
    paths = filter(p -> endswith(lowercase(p), ".ifs"), readdir(data_dir; join=true))
    sort!(paths)
    return paths
end

function list_ifs_definition_names(path::AbstractString)::Vector{String}
    defs = Fractals.parse_ifs_definitions_file(path)
    isempty(defs) && throw(ArgumentError("no definitions found in $path"))
    return [String(d.name) for d in defs]
end

function matrix_to_text(eq::AbstractMatrix{<:Real})::String
    nrows, ncols = size(eq)
    ncols == 7 || throw(ArgumentError("matrix_to_text expects exactly 7 columns"))
    lines = Vector{String}(undef, nrows)
    for i in 1:nrows
        lines[i] = @sprintf("%.10g %.10g %.10g %.10g %.10g %.10g %.10g",
                            eq[i, 1], eq[i, 2], eq[i, 3], eq[i, 4], eq[i, 5], eq[i, 6], eq[i, 7])
    end
    return join(lines, '\n')
end

function parse_matrix_text(text::AbstractString)::Matrix{Float64}
    rows = Vector{Vector{Float64}}()
    for raw in split(text, '\n')
        line = strip(raw)
        isempty(line) && continue
        parts = split(line)
        length(parts) == 7 || throw(ArgumentError("each row must have exactly 7 columns"))
        row = Vector{Float64}(undef, 7)
        for i in 1:7
            parsed = tryparse(Float64, parts[i])
            parsed === nothing && throw(ArgumentError("invalid numeric token '$(parts[i])'"))
            row[i] = parsed
        end
        push!(rows, row)
    end
    isempty(rows) && throw(ArgumentError("matrix text is empty"))

    out = Matrix{Float64}(undef, length(rows), 7)
    for i in eachindex(rows)
        out[i, :] .= rows[i]
    end
    return out
end

function _normalize_probability_column(eq::AbstractMatrix{<:Real})::Matrix{Float64}
    size(eq, 2) == 7 || throw(ArgumentError("matrix must have exactly 7 columns"))
    out = Matrix{Float64}(eq)
    probs = out[:, 7]
    all(isfinite, probs) || throw(ArgumentError("probability column contains non-finite values"))
    all(p -> p >= 0.0, probs) || throw(ArgumentError("probability column must be nonnegative"))
    total = sum(probs)
    total > 0.0 || throw(ArgumentError("probability column must have positive total weight"))
    out[:, 7] ./= total
    return out
end

function _refresh_placeholder!(state::GUIState)
    source = isnothing(state.source_file) ? "unsaved template" : basename(state.source_file)
    status = state.is_valid ? "ready to render" : "invalid matrix"
    state.fractal_placeholder_text = "Fractal preview not yet implemented\nsource: $source\ndefinition: $(state.definition_name)\nstatus: $status"
end

function _refresh_svg!(state::GUIState; width::Int=520, height::Int=520)
    requested = tempname() * ".svg"
    actual = Fractals.render_transformations_svg(state.ifs; outpath=requested, width=width, height=height)
    state.svg_temp_path = actual
    state.svg_string = read(actual, String)
    return actual
end

function make_default_state(; data_dir::AbstractString=joinpath("Fractals.jl", "data"))::GUIState
    eq = copy(_DEFAULT_ROW)
    ifs = IFS(eq; npoints=20_000, name="Template", docs="FractalsGUI template")
    state = GUIState(
        nothing,
        1,
        "Template",
        eq,
        copy(eq),
        "FractalsGUI template",
        ifs,
        "",
        "",
        "",
        nothing,
        true,
    )
    _refresh_svg!(state)
    _refresh_placeholder!(state)
    return state
end

function apply_eq_matrix!(state::GUIState, eq::AbstractMatrix{<:Real})::Bool
    normalized = try
        normalize_eq_matrix(eq)
    catch err
        state.is_valid = false
        state.last_error = sprint(showerror, err)
        _refresh_placeholder!(state)
        return false
    end

    try
        state.eq_matrix = normalized
        state.ifs = IFS(normalized; npoints=20_000, name=state.definition_name, docs=state.docs)
        state.last_error = nothing
        state.is_valid = true
        _refresh_svg!(state)
        _refresh_placeholder!(state)
        return true
    catch err
        state.is_valid = false
        state.last_error = sprint(showerror, err)
        _refresh_placeholder!(state)
        return false
    end
end

function _load_definition(path::AbstractString; definition_index::Int=1)
    defs = Fractals.parse_ifs_definitions_file(path)
    isempty(defs) && throw(ArgumentError("no definitions found in $path"))
    (1 <= definition_index <= length(defs)) || throw(ArgumentError("definition index $definition_index out of range 1:$(length(defs))"))
    selected = defs[definition_index]
    eq = normalize_eq_matrix(selected.eq)
    names = [String(d.name) for d in defs]
    return (eq=eq, docs=String(selected.docs), name=String(selected.name), names=names)
end

function _resolve_definition_index(defs;
                                   definition_index::Union{Nothing,Int}=nothing,
                                   definition_name::Union{Nothing,AbstractString}=nothing)::Int
    if !isnothing(definition_index) && !isnothing(definition_name)
        throw(ArgumentError("provide only one of definition_index or definition_name"))
    end

    if !isnothing(definition_name)
        idx = findfirst(d -> String(d.name) == String(definition_name), defs)
        isnothing(idx) && throw(ArgumentError("definition name '$(definition_name)' not found"))
        return idx
    end

    if !isnothing(definition_index)
        (1 <= definition_index <= length(defs)) || throw(ArgumentError("definition index $definition_index out of range 1:$(length(defs))"))
        return definition_index
    end

    return 1
end

function parse_ifs_definition_for_gui(path::AbstractString;
                                      definition_index::Union{Nothing,Int}=nothing,
                                      definition_name::Union{Nothing,AbstractString}=nothing)
    defs = Fractals.parse_ifs_definitions_file(path)
    isempty(defs) && throw(ArgumentError("no definitions found in $path"))
    idx = _resolve_definition_index(defs;
                                    definition_index=definition_index,
                                    definition_name=definition_name)
    selected = defs[idx]
    eq = normalize_eq_matrix(selected.eq)
    names = [String(d.name) for d in defs]
    return (eq=eq,
            docs=String(selected.docs),
            name=String(selected.name),
            names=names,
            definition_index=idx)
end

function load_ifs_definition(path::AbstractString; definition_index::Int=1)
    return _load_definition(path; definition_index=definition_index)
end

function load_ifs_definition!(state::GUIState, path::AbstractString; definition_index::Int=1)::Bool
    loaded = _load_definition(path; definition_index=definition_index)
    state.source_file = abspath(path)
    state.definition_index = definition_index
    state.definition_name = loaded.name
    state.docs = loaded.docs
    state.eq_matrix = loaded.eq
    state.loaded_eq_matrix = copy(loaded.eq)
    return apply_eq_matrix!(state, loaded.eq)
end

function load_ifs_definition_gui!(state::GUIState, path::AbstractString;
                                  definition_index::Union{Nothing,Int}=nothing,
                                  definition_name::Union{Nothing,AbstractString}=nothing)::Bool
    loaded = parse_ifs_definition_for_gui(path;
                                          definition_index=definition_index,
                                          definition_name=definition_name)
    state.source_file = abspath(path)
    state.definition_index = loaded.definition_index
    state.definition_name = loaded.name
    state.docs = loaded.docs
    state.eq_matrix = loaded.eq
    state.loaded_eq_matrix = copy(loaded.eq)
    return apply_eq_matrix!(state, loaded.eq)
end

function _make_numeric_entry(v::Real; width_chars::Int=7)
    e = Gtk.GtkEntry()
    Gtk.set_gtk_property!(e, :width_chars, width_chars)
    Gtk.set_gtk_property!(e, :text, @sprintf("%.10g", Float64(v)))
    return e
end

function _entry_float(e)::Float64
    txt = strip(Gtk.get_gtk_property(e, :text, String))
    val = tryparse(Float64, txt)
    val === nothing && throw(ArgumentError("invalid numeric token '$txt'"))
    return val
end

function _set_entry_float!(e, v::Real)
    Gtk.set_gtk_property!(e, :text, @sprintf("%.10g", Float64(v)))
end

function _make_math_label(text::AbstractString; width_chars::Union{Nothing,Integer}=nothing)
    label = Gtk.GtkLabel("")
    escaped = replace(String(text), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
    Gtk.set_gtk_property!(label, :use_markup, true)
    Gtk.set_gtk_property!(label, :label, "<tt>$escaped</tt>")
    Gtk.set_gtk_property!(label, :xalign, 0.0)
    isnothing(width_chars) || Gtk.set_gtk_property!(label, :width_chars, Int(width_chars))
    return label
end

function _function_color_hex(i::Int, n::Int)::String
    colors = Fractals._map_colors(n)
    return Fractals._rgb_to_hex(Fractals._map_color_rgb(i, colors))
end

function _make_function_color_icon(i::Int, n::Int)
    icon = Gtk.GtkLabel("")
    Gtk.set_gtk_property!(icon, :use_markup, true)
    Gtk.set_gtk_property!(icon, :label,
                          "<span foreground=\"$(_function_color_hex(i, n))\" size=\"x-large\">●</span>")
    return icon
end

function _build_function_row(eq::AbstractMatrix{<:Real}, i::Int)
    a11 = _make_numeric_entry(eq[i, 1])
    a12 = _make_numeric_entry(eq[i, 2])
    a21 = _make_numeric_entry(eq[i, 3])
    a22 = _make_numeric_entry(eq[i, 4])
    b1 = _make_numeric_entry(eq[i, 5])
    b2 = _make_numeric_entry(eq[i, 6])
    p = _make_numeric_entry(eq[i, 7]; width_chars=8)

    color_icon = _make_function_color_icon(i, size(eq, 1))
    idx_lbl = Gtk.GtkLabel("$(i)")

    row_box = Gtk.GtkBox(:h)
    eq_box = Gtk.GtkBox(:v)
    l1 = Gtk.GtkBox(:h)
    l2 = Gtk.GtkBox(:h)
    Gtk.set_gtk_property!(row_box, :spacing, 6)
    Gtk.set_gtk_property!(eq_box, :spacing, 0)
    Gtk.set_gtk_property!(l1, :spacing, 3)
    Gtk.set_gtk_property!(l2, :spacing, 3)

    Gtk.push!(l1, _make_math_label("A_i"; width_chars=3))
    Gtk.push!(l1, _make_math_label("⎡"))
    Gtk.push!(l1, a11)
    Gtk.push!(l1, a12)
    Gtk.push!(l1, _make_math_label("⎤"))

    Gtk.push!(l2, _make_math_label("" ; width_chars=3))
    Gtk.push!(l2, _make_math_label("⎣"))
    Gtk.push!(l2, a21)
    Gtk.push!(l2, a22)
    Gtk.push!(l2, _make_math_label("⎦"))

    Gtk.push!(l1, _make_math_label("" ; width_chars=2))
    Gtk.push!(l1, _make_math_label("⎡"))
    Gtk.push!(l1, _make_math_label("x"))
    Gtk.push!(l1, _make_math_label("⎤"))
    Gtk.push!(l2, _make_math_label("" ; width_chars=2))
    Gtk.push!(l2, _make_math_label("⎣"))
    Gtk.push!(l2, _make_math_label("y"))
    Gtk.push!(l2, _make_math_label("⎦"))

    Gtk.push!(l1, _make_math_label("+ b_i"; width_chars=6))
    Gtk.push!(l1, _make_math_label("⎡"))
    Gtk.push!(l1, b1)
    Gtk.push!(l1, _make_math_label("⎤"))
    Gtk.push!(l2, _make_math_label("" ; width_chars=6))
    Gtk.push!(l2, _make_math_label("⎣"))
    Gtk.push!(l2, b2)
    Gtk.push!(l2, _make_math_label("⎦"))

    Gtk.push!(l1, _make_math_label("p_i"; width_chars=4))
    Gtk.push!(l1, p)

    Gtk.push!(eq_box, l1)
    Gtk.push!(eq_box, l2)

    Gtk.push!(row_box, color_icon)
    Gtk.push!(row_box, idx_lbl)
    Gtk.push!(row_box, Gtk.GtkLabel("  "))
    Gtk.push!(row_box, eq_box)

    entries = [a11, a12, a21, a22, b1, b2, p]
    return (widget=row_box, entries=entries, color_icon=color_icon)
end

function launch_gui(; data_dir::AbstractString=joinpath("Fractals.jl", "data"))
    state = make_default_state(; data_dir=data_dir)

    window = Gtk.GtkWindow("FractalsGUI", 1580, 860)
    root = Gtk.GtkBox(:v)
    Gtk.set_gtk_property!(root, :hexpand, true)
    Gtk.set_gtk_property!(root, :vexpand, true)
    Gtk.push!(window, root)

    # Menu bar
    menubar = Gtk.GtkMenuBar()
    Gtk.push!(root, menubar)

    file_item = Gtk.GtkMenuItem("File")
    file_menu = Gtk.GtkMenu()
    Gtk.set_gtk_property!(file_item, :submenu, file_menu)
    Gtk.push!(menubar, file_item)

    load_data_item = Gtk.GtkMenuItem("Load from data/")
    load_data_menu = Gtk.GtkMenu()
    Gtk.set_gtk_property!(load_data_item, :submenu, load_data_menu)
    Gtk.push!(file_menu, load_data_item)

    definitions_item = Gtk.GtkMenuItem("IFS definitions")
    definitions_menu = Gtk.GtkMenu()
    Gtk.set_gtk_property!(definitions_item, :submenu, definitions_menu)
    Gtk.push!(file_menu, definitions_item)

    open_item = Gtk.GtkMenuItem("Open .ifs...")
    reload_item = Gtk.GtkMenuItem("Reload current source")
    clear_item = Gtk.GtkMenuItem("Clear to empty template")
    Gtk.push!(file_menu, open_item)
    Gtk.push!(file_menu, reload_item)
    Gtk.push!(file_menu, clear_item)

    # Main 3-panel row
    row = Gtk.GtkBox(:h)
    Gtk.set_gtk_property!(row, :hexpand, true)
    Gtk.set_gtk_property!(row, :vexpand, true)
    Gtk.set_gtk_property!(row, :homogeneous, true)
    Gtk.push!(root, row)

    panel_matrix = Gtk.GtkBox(:v)
    panel_svg = Gtk.GtkBox(:v)
    panel_fractal = Gtk.GtkBox(:v)
    Gtk.set_gtk_property!(panel_matrix, :hexpand, true)
    Gtk.set_gtk_property!(panel_matrix, :vexpand, true)
    Gtk.set_gtk_property!(panel_svg, :hexpand, true)
    Gtk.set_gtk_property!(panel_svg, :vexpand, true)
    Gtk.set_gtk_property!(panel_fractal, :hexpand, true)
    Gtk.set_gtk_property!(panel_fractal, :vexpand, true)

    Gtk.push!(row, panel_matrix)
    Gtk.push!(row, panel_svg)
    Gtk.push!(row, panel_fractal)

    Gtk.push!(panel_matrix, Gtk.GtkLabel("Functions as affine transforms: f_i([x;y]) = A_i*[x;y] + b_i"))
    Gtk.push!(panel_matrix, Gtk.GtkLabel("Left color icon shows which rendered color belongs to each function. Edit each numeric field directly."))
    Gtk.push!(panel_svg, Gtk.GtkLabel("Transformation SVG Preview"))
    Gtk.push!(panel_fractal, Gtk.GtkLabel("Fractal Panel (placeholder)"))

    rows_box = Gtk.GtkBox(:v)
    matrix_scroll = Gtk.GtkScrolledWindow()
    Gtk.set_gtk_property!(matrix_scroll, :hexpand, true)
    Gtk.set_gtk_property!(matrix_scroll, :vexpand, true)
    Gtk.push!(matrix_scroll, rows_box)
    Gtk.push!(panel_matrix, matrix_scroll)

    ui_rows_ref = Ref(Vector{Any}())
    load_data_items_ref = Ref(Vector{Any}())
    definition_items_ref = Ref(Vector{Any}())

    function _clear_rows!()
        # Destroy only tracked row widgets to avoid walking mutable Gtk child lists.
        for row_ui in ui_rows_ref[]
            widget = get(row_ui, :widget, nothing)
            isnothing(widget) && continue
            Gtk.destroy(widget)
        end
        empty!(ui_rows_ref[])
    end

    function _rebuild_rows!(eq::AbstractMatrix{<:Real})
        _clear_rows!()
        for i in 1:size(eq, 1)
            row_ui = _build_function_row(eq, i)
            push!(ui_rows_ref[], row_ui)
            Gtk.push!(rows_box, row_ui.widget)
        end
        Gtk.showall(rows_box)
    end

    function _collect_eq_from_rows()::Matrix{Float64}
        nrows = length(ui_rows_ref[])
        nrows > 0 || throw(ArgumentError("matrix must have at least one row"))
        eq = Matrix{Float64}(undef, nrows, 7)
        for i in 1:nrows
            entries = ui_rows_ref[][i].entries
            for j in 1:7
                eq[i, j] = _entry_float(entries[j])
            end
        end
        return eq
    end

    function _write_eq_to_rows!(eq::AbstractMatrix{<:Real})
        if length(ui_rows_ref[]) != size(eq, 1)
            _rebuild_rows!(eq)
            return
        end
        for i in 1:size(eq, 1)
            entries = ui_rows_ref[][i].entries
            for j in 1:7
                _set_entry_float!(entries[j], eq[i, j])
            end
        end
    end

    _rebuild_rows!(state.eq_matrix)

    matrix_buttons = Gtk.GtkBox(:h)
    add_row_btn = Gtk.GtkButton("Add Row")
    remove_row_btn = Gtk.GtkButton("Remove Last Row")
    normalize_p_btn = Gtk.GtkButton("Normalize p")
    apply_btn = Gtk.GtkButton("Apply")
    reset_btn = Gtk.GtkButton("Reset to Loaded")
    Gtk.push!(matrix_buttons, add_row_btn)
    Gtk.push!(matrix_buttons, remove_row_btn)
    Gtk.push!(matrix_buttons, normalize_p_btn)
    Gtk.push!(matrix_buttons, apply_btn)
    Gtk.push!(matrix_buttons, reset_btn)
    Gtk.push!(panel_matrix, matrix_buttons)

    status_label = Gtk.GtkLabel("Ready")
    Gtk.push!(panel_matrix, status_label)

    svg_image = Gtk.GtkImage()
    Gtk.set_gtk_property!(svg_image, :file, state.svg_temp_path)
    svg_scroll = Gtk.GtkScrolledWindow()
    Gtk.set_gtk_property!(svg_scroll, :hexpand, true)
    Gtk.set_gtk_property!(svg_scroll, :vexpand, true)
    Gtk.push!(svg_scroll, svg_image)
    Gtk.push!(panel_svg, svg_scroll)

    placeholder_label = Gtk.GtkLabel(state.fractal_placeholder_text)
    Gtk.push!(panel_fractal, placeholder_label)

    function set_status!()
        if state.is_valid
            Gtk.set_gtk_property!(status_label, :label, "Ready")
        else
            Gtk.set_gtk_property!(status_label, :label, "Error: " * something(state.last_error, "unknown error"))
        end
        Gtk.set_gtk_property!(placeholder_label, :label, state.fractal_placeholder_text)
        Gtk.set_gtk_property!(svg_image, :file, state.svg_temp_path)
    end

    function apply_editor_values!()
        parsed = try
            _collect_eq_from_rows()
        catch err
            state.is_valid = false
            state.last_error = sprint(showerror, err)
            _refresh_placeholder!(state)
            set_status!()
            return
        end
        if apply_eq_matrix!(state, parsed)
            state.loaded_eq_matrix = copy(state.eq_matrix)
            _write_eq_to_rows!(state.eq_matrix)
        end
        set_status!()
    end

    function reload_data_menu!()
        for item in load_data_items_ref[]
            isnothing(item) && continue
            Gtk.destroy(item)
        end
        empty!(load_data_items_ref[])
        files = list_data_ifs(data_dir)
        if isempty(files)
            empty_item = Gtk.GtkMenuItem("(no .ifs files found)")
            Gtk.push!(load_data_menu, empty_item)
            push!(load_data_items_ref[], empty_item)
            return
        end

        for file in files
            item = Gtk.GtkMenuItem(basename(file))
            Gtk.signal_connect(item, "activate") do _
                try
                    ok = load_ifs_definition_gui!(state, file)
                    if ok
                        _rebuild_rows!(state.eq_matrix)
                    end
                    reload_definitions_menu!()
                    set_status!()
                catch err
                    Gtk.set_gtk_property!(status_label, :label, "Error: " * sprint(showerror, err))
                end
            end
            Gtk.push!(load_data_menu, item)
            push!(load_data_items_ref[], item)
        end
    end

    function reload_definitions_menu!()
        for item in definition_items_ref[]
            isnothing(item) && continue
            Gtk.destroy(item)
        end
        empty!(definition_items_ref[])

        if isnothing(state.source_file)
            empty_item = Gtk.GtkMenuItem("(load a file first)")
            Gtk.set_gtk_property!(empty_item, :sensitive, false)
            Gtk.push!(definitions_menu, empty_item)
            push!(definition_items_ref[], empty_item)
            Gtk.showall(definitions_menu)
            return
        end

        names = try
            list_ifs_definition_names(state.source_file)
        catch err
            empty_item = Gtk.GtkMenuItem("(failed to parse definitions)")
            Gtk.set_gtk_property!(empty_item, :sensitive, false)
            Gtk.push!(definitions_menu, empty_item)
            push!(definition_items_ref[], empty_item)
            Gtk.set_gtk_property!(status_label, :label, "Error: " * sprint(showerror, err))
            Gtk.showall(definitions_menu)
            return
        end

        for (i, name) in enumerate(names)
            label = "[$i] $name"
            item = Gtk.GtkMenuItem(label)
            Gtk.set_gtk_property!(item, :sensitive, i != state.definition_index)
            Gtk.signal_connect(item, "activate") do _
                try
                    ok = load_ifs_definition_gui!(state, state.source_file; definition_index=i)
                    if ok
                        _rebuild_rows!(state.eq_matrix)
                    end
                    reload_definitions_menu!()
                    set_status!()
                catch err
                    Gtk.set_gtk_property!(status_label, :label, "Error: " * sprint(showerror, err))
                end
            end
            Gtk.push!(definitions_menu, item)
            push!(definition_items_ref[], item)
        end
        Gtk.showall(definitions_menu)
    end

    Gtk.signal_connect(add_row_btn, "clicked") do _
        try
            current = try
                _collect_eq_from_rows()
            catch
                state.eq_matrix
            end
            next = vcat(current, _DEFAULT_ROW)
            _rebuild_rows!(next)
            Gtk.set_gtk_property!(status_label, :label, "Ready")
        catch err
            Gtk.set_gtk_property!(status_label, :label, "Error: " * sprint(showerror, err))
        end
    end

    Gtk.signal_connect(remove_row_btn, "clicked") do _
        current = try
            _collect_eq_from_rows()
        catch
            state.eq_matrix
        end
        if size(current, 1) <= 1
            Gtk.set_gtk_property!(status_label, :label, "Error: matrix must keep at least one row")
            return
        end
        _rebuild_rows!(current[1:end-1, :])
    end

    Gtk.signal_connect(apply_btn, "clicked") do _
        apply_editor_values!()
    end

    Gtk.signal_connect(normalize_p_btn, "clicked") do _
        parsed = try
            _collect_eq_from_rows()
        catch err
            state.is_valid = false
            state.last_error = sprint(showerror, err)
            _refresh_placeholder!(state)
            set_status!()
            return
        end
        normalized = try
            _normalize_probability_column(parsed)
        catch err
            state.is_valid = false
            state.last_error = sprint(showerror, err)
            _refresh_placeholder!(state)
            set_status!()
            return
        end
        _write_eq_to_rows!(normalized)
        apply_editor_values!()
    end

    Gtk.signal_connect(reset_btn, "clicked") do _
        _rebuild_rows!(state.loaded_eq_matrix)
        _ = apply_eq_matrix!(state, state.loaded_eq_matrix)
        set_status!()
    end

    Gtk.signal_connect(open_item, "activate") do _
        chosen = try
            Gtk.open_dialog("Open .ifs file", window, ("IFS files", "*.ifs"))
        catch err
            Gtk.set_gtk_property!(status_label, :label, "Error: " * sprint(showerror, err))
            return
        end
        isnothing(chosen) && return
        isempty(strip(chosen)) && return
        ok = load_ifs_definition_gui!(state, chosen)
        if ok
            _rebuild_rows!(state.eq_matrix)
        end
        reload_definitions_menu!()
        set_status!()
    end

    Gtk.signal_connect(reload_item, "activate") do _
        if isnothing(state.source_file)
            Gtk.set_gtk_property!(status_label, :label, "No source file loaded")
            return
        end
        ok = load_ifs_definition_gui!(state, state.source_file; definition_index=state.definition_index)
        if ok
            _rebuild_rows!(state.eq_matrix)
        end
        reload_definitions_menu!()
        set_status!()
    end

    Gtk.signal_connect(clear_item, "activate") do _
        state.source_file = nothing
        state.definition_index = 1
        state.definition_name = "Template"
        state.docs = "FractalsGUI template"
        state.loaded_eq_matrix = copy(_DEFAULT_ROW)
        _ = apply_eq_matrix!(state, state.loaded_eq_matrix)
        _rebuild_rows!(state.eq_matrix)
        reload_definitions_menu!()
        set_status!()
    end

    Gtk.signal_connect(window, "destroy") do _
        Gtk.gtk_quit()
    end

    reload_data_menu!()
    reload_definitions_menu!()
    set_status!()
    Gtk.showall(window)
    Gtk.maximize(window)
    Gtk.gtk_main()
    return nothing
end

end
