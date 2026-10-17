using Test
using IFSParse
using IFSParse:
    _noncontractive_rows,
    _format_contractivity_failure,
    _split_valid_ifs_definitions,
    _validated_ifs_definitions,
    _eq_weights,
    _ifs_key

const JLD2 = IFSParse.JLD2

# Every filesystem test works in a temp dir with `dir`/`outdir`/`libdir` passed explicitly, so
# nothing is written into the repository.
function with_tmp(f)
    dir = mktempdir()
    try
        return f(dir)
    finally
        rm(dir; recursive = true, force = true)
    end
end

const ALPHA_ROWS = """
  0.5 0.0 0.0 0.5 0.0 0.0 0.5
  0.5 0.0 0.0 0.5 0.5 0.0 0.5
"""

# Two source files: a.ifs holds Alpha and Beta, b.ifs holds Gamma and a non-contractive Bad.
function write_fixtures(dir)
    mkpath(dir)
    write(
        joinpath(dir, "a.ifs"),
        "Alpha {\n; alpha docs\n$ALPHA_ROWS}\nBeta {\n  0.4 0.0 0.0 0.4 0.0 0.0\n  0.4 0.0 0.0 0.4 0.6 0.0\n}\n",
    )
    write(
        joinpath(dir, "b.ifs"),
        "Gamma {\n  0.3 0.1 -0.1 0.3 0.2 0.1 1.0\n}\nBad {\n  1.0 0.0 0.0 1.0 0.0 0.0 1.0\n}\n",
    )
    return dir
end

# The one non-contractive definition warns on every build/parse of b.ifs.
const BAD_WARN = r"Skipping non-contractive IFS definition 'Bad'"
build(; kwargs...) = @test_logs (:warn, BAD_WARN) build_ifs_library(; kwargs...)

@testset "IFSParse" begin
    @testset "lex_ifs tokenizes IFS text into a structured token stream" begin
        input = """
        MyFractal {; Some documentation
        ; More docs
          0.5 0.0 0.0 0.5 0.0 0.0 0.6
         -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
        }
        """
        tokens = lex_ifs(input)
        @test all(t -> t isa IFSToken, tokens)
        kinds = [t.kind for t in tokens]
        for k in (:NAME, :LBRACE, :DOCS, :ARRAY, :RBRACE)
            @test k in kinds
        end

        name_tok = first(t for t in tokens if t.kind == :NAME)
        @test name_tok.value == "MyFractal"

        array_toks = filter(t -> t.kind == :ARRAY, tokens)
        @test length(array_toks) == 2
        @test length(array_toks[1].value) == 7
        @test array_toks[1].value[1] ≈ 0.5

        # Lines containing (3D) are skipped.
        @test isempty(lex_ifs("Skipped (3D) {\n"))
    end

    @testset "parse_ifs_definitions_string extracts name, docs, and equation matrices" begin
        sample = """
        Test IFS {; Doc line 1
        ; Doc line 2
          0.5 0.0 0.0 0.5 0.0 0.0 0.6
         -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
        }
        """
        defs = parse_ifs_definitions_string(sample)
        @test length(defs) == 1
        @test defs[1] isa IFSDefinition
        @test defs[1].name == "Test IFS"
        @test occursin("Doc line 1", defs[1].docs)
        @test occursin("Doc line 2", defs[1].docs)
        @test size(defs[1].eq) == (2, 7)
        @test defs[1].eq[2, 1] == -0.5

        # Several blocks in one string.
        two = parse_ifs_definitions_string(sample * sample)
        @test length(two) == 2
    end

    @testset "parse_ifs_definitions_file reads blocks from a file" begin
        with_tmp() do tmp
            path = joinpath(tmp, "file.ifs")
            write(path, "File IFS {\n$ALPHA_ROWS}\n")
            defs = parse_ifs_definitions_file(path)
            @test length(defs) == 1
            @test defs[1].name == "File IFS"
            @test size(defs[1].eq) == (2, 7)
        end
    end

    @testset "negative parser cases" begin
        missing_close = """
        BrokenIFS {
          0.5 0.0 0.0 0.5 0.0 0.0
        """
        @test_throws EOFError parse_ifs_definitions_string(missing_close)

        stray_close = """
        }
        ValidName {
          0.5 0.0 0.0 0.5 0.0 0.0
        }
        """
        err = try
            parse_ifs_definitions_string(stray_close)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Unexpected '}' outside IFS block", sprint(showerror, err))

        mixed_width = """
        MixedWidth {
          0.5 0.0 0.0 0.5 0.0 0.0
          0.5 0.0 0.0 0.5 0.5 0.0 0.5
        }
        """
        err = try
            parse_ifs_definitions_string(mixed_width)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("has length", sprint(showerror, err))

        bad_float = """
        BadFloat {
          0.5 0.0 abc 0.5 0.0 0.0
        }
        """
        err = try
            parse_ifs_definitions_string(bad_float)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Invalid numeric token 'abc'", sprint(showerror, err))

        @test_throws ArgumentError parse_ifs_definitions_string("{\n 0.5 0 0 0.5 0 0\n}\n")
        @test_throws ArgumentError parse_ifs_definitions_string("Empty {\n}\n")

        with_tmp() do tmp
            path = joinpath(tmp, "mixed.ifs")
            write(path, mixed_width)
            @test_throws ArgumentError parse_ifs_definitions_file(path)
        end
    end

    @testset "weights follow the 7th column or |det(A)|" begin
        @test _eq_weights([0.5 0.0 0.0 0.5 0.0 0.0 0.7]) == [0.7]
        @test _eq_weights([0.5 0.0 0.0 0.4 0.0 0.0; 0.0 -0.5 0.5 0.0 0.0 0.0]) ≈ [0.2, 0.25]
    end

    @testset "contractivity validation and skipping" begin
        bad_eq = [1.0 0.0 0.0 1.0 0.0 0.0 1.0]
        failures = _noncontractive_rows(bad_eq)
        @test length(failures) == 1
        @test failures[1].row == 1
        @test occursin("row 1", _format_contractivity_failure(failures[1]))
        @test isempty(_noncontractive_rows([0.5 0.0 0.0 0.5 0.0 0.0]))

        mixed = """
        ValidOne {
          0.5 0.0 0.0 0.5 0.0 0.0 1.0
        }
        InvalidOne {
          1.0 0.0 0.0 1.0 0.0 0.0 1.0
        }
        ValidTwo {
          0.3 0.1 -0.1 0.3 0.2 0.1 1.0
        }
        """
        defs = parse_ifs_definitions_string(mixed)
        @test length(defs) == 3  # the raw parser keeps every block

        valid, invalid = _split_valid_ifs_definitions(defs)
        @test [d.name for d in valid] == ["ValidOne", "ValidTwo"]
        @test [e.definition.name for e in invalid] == ["InvalidOne"]

        kept =
            @test_logs (:warn, r"Skipping non-contractive IFS definition 'InvalidOne'") _validated_ifs_definitions(
                defs;
                source = "mixed",
            )
        @test [d.name for d in kept] == ["ValidOne", "ValidTwo"]
    end

    @testset "ifs_data_dir points at the repo-root ifs/ directory" begin
        @test ifs_data_dir() == joinpath(dirname(pkgdir(IFSParse)), "ifs")
        @test isfile(ifs_data_dir("Default.ifs"))
        @test !isempty(parse_ifs_definitions_file(ifs_data_dir("Default.ifs")))
        names = basename.(list_ifs_files(; dir = ifs_data_dir()))
        @test "Default.ifs" in names
        @test "Edgar.ifs" in names
        @test issorted(names)
        @test isempty(list_ifs_files(; dir = joinpath(ifs_data_dir(), "no_such_dir")))
    end

    @testset "list_ifs_files finds the .ifs files in a directory" begin
        with_tmp() do tmp
            write_fixtures(joinpath(tmp, "src"))
            write(joinpath(tmp, "src", "notes.txt"), "not an ifs file")
            files = list_ifs_files(; dir = joinpath(tmp, "src"))
            @test basename.(files) == ["a.ifs", "b.ifs"]
            @test all(isfile, files)
        end
    end

    @testset "build_ifs_library writes one record per definition" begin
        with_tmp() do tmp
            src = write_fixtures(joinpath(tmp, "src"))
            lib = joinpath(tmp, "lib")
            paths = build(; dir = src, outdir = lib)

            # Alpha and Beta from a.ifs, Gamma from b.ifs; Bad was skipped with a warning.
            @test length(paths) == 3
            @test length(unique(paths)) == 3
            @test all(isfile, paths)
            @test sort(readdir(lib)) == ["a", "b"]
            @test length(readdir(joinpath(lib, "a"))) == 2
            @test length(readdir(joinpath(lib, "b"))) == 1
            for p in paths
                @test occursin(r"^ifs_[0-9a-f]{16}\.jld2$", basename(p))
                @test dirname(dirname(p)) == lib
            end

            for p in paths
                rec = JLD2.load(p)
                @test sort(collect(keys(rec))) ==
                      ["docs", "eq", "key", "name", "source_file", "weights"]
                @test basename(p) == "ifs_" * rec["key"] * ".jld2"
                @test rec["key"] == _ifs_key(rec["eq"], rec["weights"])
                @test rec["source_file"] == basename(dirname(p)) * ".ifs"
                @test length(rec["weights"]) == size(rec["eq"], 1)
            end
            @test sort([JLD2.load(p)["name"] for p in paths]) == ["Alpha", "Beta", "Gamma"]
        end
    end

    @testset "rebuilding reuses existing records" begin
        with_tmp() do tmp
            src = write_fixtures(joinpath(tmp, "src"))
            lib = joinpath(tmp, "lib")
            first_paths = build(; dir = src, outdir = lib)
            mtimes = [mtime(p) for p in first_paths]
            second_paths = build(; dir = src, outdir = lib)
            @test second_paths == first_paths
            @test [mtime(p) for p in second_paths] == mtimes

            # force = true rewrites to the same paths.
            forced = build(; dir = src, outdir = lib, force = true)
            @test forced == first_paths
        end
    end

    @testset "list_ifs covers prebuilt and .ifs sources, optionally grouped by file" begin
        with_tmp() do tmp
            src = write_fixtures(joinpath(tmp, "src"))
            lib = joinpath(tmp, "lib")

            # Nothing prebuilt yet: everything comes from the .ifs files (Bad is left out).
            entries = list_ifs(; dir = src, libdir = lib)
            @test [e.name for e in entries] == ["Alpha", "Beta", "Gamma"]
            @test all(e -> e.source == :ifs_file, entries)
            @test [e.file for e in entries] == ["a.ifs", "a.ifs", "b.ifs"]
            @test all(e -> isfile(e.path), entries)
            @test keys(first(entries)) == (:name, :file, :source, :path)

            build(; dir = src, outdir = lib)
            entries = list_ifs(; dir = src, libdir = lib)
            @test [e.name for e in entries] == ["Alpha", "Beta", "Gamma"]
            @test all(e -> e.source == :prebuilt, entries)  # listed once, as prebuilt
            @test all(e -> endswith(e.path, ".jld2"), entries)

            # A definition added after the build only exists in its .ifs file.
            write(joinpath(src, "c.ifs"), "Delta {\n$ALPHA_ROWS}\n")
            entries = list_ifs(; dir = src, libdir = lib)
            @test [e.name for e in entries] == ["Alpha", "Beta", "Gamma", "Delta"]
            @test [e.source for e in entries] == [:prebuilt, :prebuilt, :prebuilt, :ifs_file]

            grouped = list_ifs(; dir = src, libdir = lib, group_by_file = true)
            @test first.(grouped) == ["a.ifs", "b.ifs", "c.ifs"]
            @test [length(last(g)) for g in grouped] == [2, 1, 1]

            # Missing directories are simply empty sources.
            @test isempty(list_ifs(; dir = joinpath(tmp, "none"), libdir = joinpath(tmp, "none")))
        end
    end

    @testset "get_ifs_definition prefers prebuilt records and falls back to .ifs files" begin
        with_tmp() do tmp
            src = write_fixtures(joinpath(tmp, "src"))
            lib = joinpath(tmp, "lib")
            empty_lib = joinpath(tmp, "empty_lib")
            build(; dir = src, outdir = lib)

            d = get_ifs_definition("Alpha"; dir = src, libdir = lib)
            @test d isa IFSDefinition
            @test d.name == "Alpha"
            @test occursin("alpha docs", d.docs)
            @test size(d.eq) == (2, 7)
            @test get_ifs_definition("Alpha"; file = "a.ifs", dir = src, libdir = lib).eq == d.eq

            # Restricting to the wrong source file finds nothing.
            @test_throws ArgumentError get_ifs_definition(
                "Alpha";
                file = "b.ifs",
                dir = src,
                libdir = lib,
            )
            @test_throws ArgumentError get_ifs_definition("Nope"; dir = src, libdir = lib)

            # No library: parse the .ifs file instead, with the same result.
            parsed = get_ifs_definition("Alpha"; dir = src, libdir = empty_lib)
            @test parsed.name == d.name
            @test parsed.docs == d.docs
            @test parsed.eq == d.eq

            # Preference: after Alpha is renamed in its source file, the prebuilt record still
            # answers to the old name, and the new name is found by parsing the file.
            write(joinpath(src, "a.ifs"), "AlphaRenamed {\n; alpha docs\n$ALPHA_ROWS}\n")
            @test get_ifs_definition("Alpha"; dir = src, libdir = lib).name == "Alpha"
            @test get_ifs_definition("AlphaRenamed"; dir = src, libdir = lib).eq == d.eq
            @test_throws ArgumentError get_ifs_definition("Alpha"; dir = src, libdir = empty_lib)

            # Non-contractive definitions are never returned.
            @test_logs (:warn, BAD_WARN) @test_throws ArgumentError get_ifs_definition(
                "Bad";
                dir = src,
                libdir = empty_lib,
            )
        end
    end

    @testset "identical systems with different names share one record" begin
        with_tmp() do tmp
            src = joinpath(tmp, "src")
            mkpath(src)
            write(joinpath(src, "twins.ifs"), "Alpha {\n$ALPHA_ROWS}\nAlphaTwin {\n$ALPHA_ROWS}\n")
            lib = joinpath(tmp, "lib")
            paths = build_ifs_library(; dir = src, outdir = lib)
            @test length(paths) == 2
            @test length(unique(paths)) == 1
            @test length(readdir(joinpath(lib, "twins"))) == 1
            # The first name wins.
            @test JLD2.load(paths[1])["name"] == "Alpha"

            # The same system in a different source file gets its own record.
            write(joinpath(src, "other.ifs"), "Alpha {\n$ALPHA_ROWS}\n")
            more = build_ifs_library(; dir = src, outdir = lib)
            @test length(unique(more)) == 2
            @test sort(readdir(lib)) == ["other", "twins"]
        end
    end

    @testset "renaming keeps the same key and the stored name" begin
        with_tmp() do tmp
            src = joinpath(tmp, "src")
            mkpath(src)
            lib = joinpath(tmp, "lib")
            write(joinpath(src, "t.ifs"), "Before {\n$ALPHA_ROWS}\n")
            before = only(build_ifs_library(; dir = src, outdir = lib))
            write(joinpath(src, "t.ifs"), "After {\n$ALPHA_ROWS}\n")
            after = only(build_ifs_library(; dir = src, outdir = lib))

            @test after == before
            @test length(readdir(joinpath(lib, "t"))) == 1
            @test JLD2.load(after)["name"] == "Before"

            # Changing the maps does change the key (and creates a second record).
            write(joinpath(src, "t.ifs"), "After {\n 0.5 0.0 0.0 0.5 0.0 0.0 0.9\n}\n")
            changed = only(build_ifs_library(; dir = src, outdir = lib))
            @test changed != before
            @test length(readdir(joinpath(lib, "t"))) == 2

            # Only the numbers enter the key (and -0.0 equals 0.0).
            eq = [0.5 0.0 0.0 0.5 0.0 0.0 0.5]
            @test _ifs_key(eq, [0.5]) == _ifs_key([0.5 -0.0 -0.0 0.5 -0.0 -0.0 0.5], [0.5])
            @test _ifs_key(eq, [0.5]) != _ifs_key(eq, [0.6])
        end
    end

    @testset "the content key is stable across sessions" begin
        with_tmp() do tmp
            src = joinpath(tmp, "src")
            mkpath(src)
            cp(ifs_data_dir("Default.ifs"), joinpath(src, "Default.ifs"))
            paths = build_ifs_library(; dir = src, outdir = joinpath(tmp, "lib"))
            sierpinski = only(paths)
            @test JLD2.load(sierpinski)["name"] == "Sierpinski"
            @test basename(sierpinski) == "ifs_e20bf0a06df4f9dc.jld2"
            @test basename(dirname(sierpinski)) == "Default"
        end
    end
end
