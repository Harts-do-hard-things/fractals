# CLI

Entry point:

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl <command> [options]
```

## Commands

### `render`

Render one fractal from an `.ifs` file or matrix text file.

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl render --input my_fractals.ifs --ifs-name "Heighway Dragon" --method Chaos --npoints 200000 --resolution 1024x1024 --out media/cli_render.png
```

### `batch-render`

Render all definitions from one `.ifs` file.

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl batch-render --input my_fractals.ifs --npoints 100000 --resolution 800x800 --out-dir media/batch
```

### `validate-ifs`

Parse and validate `.ifs` input without rendering.

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl validate-ifs --input my_fractals.ifs
```

### `benchmark`

Run a standard benchmark set.

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl benchmark --npoints 50000 --resolution 256x256 --inverse-iterations 2
```

## Notes

- Methods accepted by CLI are the same as the API (`Chaos`, `Parallel`, `PointDeterministic`, `ImageIterate`, `Inverse`, plus symbol/string forms).
- For point-deterministic/inverse methods, use `--iterations`.
- For image-iterate method, use:
  - `--image-source polygon|chaos|point_deterministic|inverse|file`
  - `--image-path <path>` when source is `file`
  - `--image-iterations <int>`
  - `--polygon-limits-mode ifs|default`
    - For `--image-source polygon`, `default` is treated as `ifs` (compatibility alias).
  - `--initial-polygon default|equilateral_triangle|line_arrow|line`
    - Effective when `--image-source polygon`; otherwise ignored.
- For multi-definition `.ifs`, select one using `--ifs-index` or `--ifs-name`.
