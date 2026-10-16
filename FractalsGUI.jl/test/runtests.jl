using Test
using FractalsGUI
import Gtk

@testset "Matrix Normalization" begin
    eq6 = [
        0.5 0.0 0.0 0.5 0.0 0.0
        0.0 0.5 -0.5 0.0 1.0 0.0
    ]
    eq7 = normalize_eq_matrix(eq6)
    @test size(eq7) == (2, 7)
    @test all(eq7[:, 7] .>= 0)
    @test sum(eq7[:, 7]) > 0
end

@testset "Matrix Text Roundtrip" begin
    eq = [
        1.0 0.0 0.0 1.0 0.0 0.0 1.0
        0.5 0.0 0.0 0.5 1.0 0.0 0.25
    ]
    txt = matrix_to_text(eq)
    parsed = parse_matrix_text(txt)
    @test parsed == eq
end

@testset "Load Definitions By Index" begin
    sample = """
    First {
      0.5 0.0 0.0 0.5 0.0 0.0
      0.5 0.0 0.0 0.5 0.5 0.0
    }
    Second {
      0.5 -0.5 0.5 0.5 0.0 0.0 0.5
      -0.5 -0.5 0.5 -0.5 1.0 0.0 0.5
    }
    """

    mktemp() do path, io
        write(io, sample)
        close(io)

        loaded = load_ifs_definition(path; definition_index=2)
        @test loaded.name == "Second"
        @test size(loaded.eq, 2) == 7
        @test length(loaded.names) == 2
    end
end

@testset "GUI Definition Loader (Non-Interactive)" begin
    sample = """
    First {
      0.5 0.0 0.0 0.5 0.0 0.0
      0.5 0.0 0.0 0.5 0.5 0.0
    }
    Second {
      0.5 -0.5 0.5 0.5 0.0 0.0 0.5
      -0.5 -0.5 0.5 -0.5 1.0 0.0 0.5
    }
    """

    mktemp() do path, io
        write(io, sample)
        close(io)

        loaded_default = parse_ifs_definition_for_gui(path)
        @test loaded_default.definition_index == 1
        @test loaded_default.name == "First"
        @test length(loaded_default.names) == 2

        loaded_name = parse_ifs_definition_for_gui(path; definition_name="Second")
        @test loaded_name.definition_index == 2
        @test loaded_name.name == "Second"

        state = make_default_state()
        @test load_ifs_definition_gui!(state, path)
        @test state.definition_index == 1
        @test state.definition_name == "First"

        @test load_ifs_definition_gui!(state, path; definition_name="Second")
        @test state.definition_index == 2
        @test state.definition_name == "Second"
    end
end

@testset "GUI Definition Listing And Switching" begin
    sample = """
    One {
      0.5 0.0 0.0 0.5 0.0 0.0
    }
    Two {
      0.3 0.0 0.0 0.3 0.4 0.0
    }
    Three {
      0.2 0.0 0.0 0.2 0.8 0.0
    }
    """

    mktemp() do path, io
        write(io, sample)
        close(io)

        names = list_ifs_definition_names(path)
        @test names == ["One", "Two", "Three"]

        state = make_default_state()
        @test load_ifs_definition_gui!(state, path; definition_index=3)
        @test state.definition_index == 3
        @test state.definition_name == "Three"

        @test load_ifs_definition_gui!(state, path; definition_index=2)
        @test state.definition_index == 2
        @test state.definition_name == "Two"
    end
end

@testset "Data Directory Listing" begin
    mktempdir() do dir
        mkpath(joinpath(dir, "nested"))
        touch(joinpath(dir, "b_file.ifs"))
        touch(joinpath(dir, "a_file.ifs"))
        touch(joinpath(dir, "ignore.txt"))
        touch(joinpath(dir, "nested", "inner.ifs"))

        @test list_data_ifs(dir) == [
            joinpath(dir, "a_file.ifs"),
            joinpath(dir, "b_file.ifs"),
        ]
    end
end

@testset "State Apply + SVG Refresh" begin
    state = make_default_state()
    old_svg_path = state.svg_temp_path
    eq = copy(state.eq_matrix)
    eq[1, 5] += 0.2
    ok = apply_eq_matrix!(state, eq)
    @test ok
    @test state.is_valid
    @test state.last_error === nothing
    @test state.svg_temp_path != old_svg_path
    @test occursin("<svg", state.svg_string)
end

@testset "State Validation Rejection" begin
    state = make_default_state()
    bad = copy(state.eq_matrix)
    bad[1, 7] = -1.0
    ok = apply_eq_matrix!(state, bad)
    @test !ok
    @test !state.is_valid
    @test state.last_error !== nothing
end

@testset "Public Matrix Validation" begin
    eq = [
        1.0 0.0 0.0 1.0 0.0 0.0 2.0
        0.5 0.0 0.0 0.5 1.0 0.0 3.0
    ]
    out = normalize_eq_matrix(eq)
    @test out == eq

    bad = copy(eq)
    bad[1, 7] = -1.0
    @test_throws ArgumentError normalize_eq_matrix(bad)

    @test_throws ArgumentError parse_matrix_text("1 2 3")
    @test_throws ArgumentError parse_matrix_text("1 2 3 4 5 6 nope")
end

@testset "Default GUI State" begin
    state = make_default_state()
    @test state.definition_name == "Template"
    @test state.is_valid
    @test state.last_error === nothing
    @test occursin("ready to render", state.fractal_placeholder_text)
    @test occursin("Template", state.fractal_placeholder_text)
    @test isfile(state.svg_temp_path)
    @test occursin("<svg", state.svg_string)
end

@testset "Function Row Uses Render Color Icon" begin
    eq = [
        0.5 -0.5 0.5 0.5 0.0 0.0 0.5
        0.5 0.5 -0.5 0.5 0.5 0.5 0.5
    ]
    row_ui = FractalsGUI._build_function_row(eq, 1)
    label = Gtk.get_gtk_property(row_ui.color_icon, :label, String)
    @test occursin("foreground=\"", label)
    @test occursin("●", label)
    @test occursin(FractalsGUI._function_color_hex(1, 2), label)
    @test !haskey(Dict(pairs(row_ui)), :color_btn)
end
