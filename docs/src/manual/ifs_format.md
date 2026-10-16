# IFS text format

IFS definitions can be written as text: inline strings for [`parse_ifs_string`](@ref), or
`.ifs` files for [`parse_ifs_file`](@ref), [`render`](@ref), and the CLI. A curated library
ships in `data/exp_raw/`; locate it with [`ifs_data_dir`](@ref), e.g.
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
