using Test
using FractalsGUI

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

@testset "Normalize Probability Column" begin
    eq = [
        1.0 0.0 0.0 1.0 0.0 0.0 2.0
        0.5 0.0 0.0 0.5 1.0 0.0 3.0
    ]
    out = FractalsGUI._normalize_probability_column(eq)
    @test isapprox(sum(out[:, 7]), 1.0; atol=1e-12)
    @test out[1, 7] ≈ 0.4
    @test out[2, 7] ≈ 0.6

    bad = copy(eq)
    bad[1, 7] = -1.0
    @test_throws ArgumentError FractalsGUI._normalize_probability_column(bad)
end

@testset "Gtk Compatibility Guard" begin
    src_path = normpath(joinpath(@__DIR__, "..", "src", "FractalsGUI.jl"))
    src = read(src_path, String)
    @test !occursin("GAccessor.text(", src)
    @test occursin("import Gtk", src)
    @test !occursin("@eval import Gtk", src)
end
