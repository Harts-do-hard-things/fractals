# IFS text format

IFS definitions can be written as text: inline strings for [`parse_ifs_string`](@ref), or
`.ifs` files for [`parse_ifs_file`](@ref), and [`get_ifs`](@ref), which feed [`render`](@ref). A curated set
ships in `ifs/`; locate it with [`ifs_data_dir`](@ref), e.g.
`parse_ifs_file(ifs_data_dir("Default.ifs"))`.

```text
Example Fractal {
; Optional docs line
  0.5  0.0  0.0  0.5  0.0  0.0  0.6
 -0.5  0.0  0.0 -0.5  1.0  0.0  0.4
}
```

## Rules

- A block starts with `<name> {` and ends with `}`. A file may hold several blocks.
- Lines starting with `;` are documentation, kept in the definition's `docs`.
- Each numeric row is one map, `a11 a12 a21 a22 b1 b2 [p]`, meaning
  `x' = a11*x + a12*y + b1` and `y' = a21*x + a22*y + b2` (Fractint order).
- Rows within a block must all have 6 columns (weights derived from `|det(A)|`) or all
  have 7 (explicit probability `p`).
- Lines containing `(3D)` are ignored.
- Non-contractive definitions are skipped with a warning when parsed into `IFS` values.


## Where `.ifs` files live and the prebuilt library

`.ifs` source files live in `ifs/` at the repo root (`projectdir("ifs")`; [`list_ifs_files`](@ref)
lists them). Parsing them on every use is cheap, but you can prebuild a library once:

```julia
build_ifs_library()    # one JLD2 record per function system
```

[`build_ifs_library`](@ref) parses every `ifs/*.ifs` file and, through DrWatson's
`produce_or_load`, saves each definition (not each file) as its own record in
`data/exp_pro/ifs/<file stem>/ifs_<key>.jld2`. A record stores the equation matrix, weights,
name, docs and source file name. The `<key>` is a hash of the equation matrix and weights only:
the name is stored but is not part of the key, so renaming a fractal does not create a new
record and rebuilding skips records that already exist (`force = true` rewrites them).
Non-contractive definitions are skipped with a warning.

- [`list_ifs`](@ref) lists definitions from both the prebuilt records and the `.ifs` files as
  `(name, file, source, path)` named tuples (`source` is `:prebuilt` or `:ifs_file`);
  `group_by_file = true` groups them by source file. A definition present in both is listed
  once, as prebuilt.
- [`get_ifs`](@ref)`("Name"; file = "Default.ifs")` builds an [`IFS`](@ref) by name,
  preferring the prebuilt record and falling back to parsing the `.ifs` file.

`projectdir()` and `datadir()` follow the active project, so under `--project=Fractals.jl`
the default locations are inside `Fractals.jl/`; pass `dir` (and `outdir`/`libdir`) explicitly,
e.g. `build_ifs_library(; dir = ifs_data_dir())`.
## Parser errors

- A missing closing brace throws `EOFError`.
- An unexpected `}` outside a block throws `ArgumentError` with the line number.
- Mixed row widths in one block throw `ArgumentError`.
- Invalid numeric tokens throw `ArgumentError` with the token and line number.

The tokens are available directly from [`lex_ifs`](@ref).

## Matrix validation

Constructing an [`IFS`](@ref) from an equation matrix requires:

- at least one row, and exactly 6 or 7 columns;
- finite entries (`NaN`/`Inf` are rejected);
- with a 7th column: nonnegative probabilities with a positive total.

Violations throw `ArgumentError`. Map-indexed operations validate bounds and report the
valid range.
