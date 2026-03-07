# CLI

Entry point:

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl <command> [options]
```

## Commands

### `render`

Render one fractal from an `.ifs` file or matrix text file.

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl render --input my_fractals.ifs --ifs-name "Heighway Dragon" --method Chaos --npoints 200000 --resolution 1024x1024 --out media/cli_render.png
```

### `batch-render`

Render all definitions from one `.ifs` file.

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl batch-render --input my_fractals.ifs --npoints 100000 --resolution 800x800 --out-dir media/batch
```

### `validate-ifs`

Parse and validate `.ifs` input without rendering.

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl validate-ifs --input my_fractals.ifs
```

### `benchmark`

Run a standard benchmark set.

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --npoints 50000 --resolution 256x256 --inverse-iterations 2
```

## Render Methods (CLI)

Methods are listed in canonical order: `Chaos`, `PointDeterministic`, `ImageIterate`, `Inverse`.
`Parallel` is a compatibility alias for `Chaos` and may be deprecated in a future release.

### `Chaos`

Arguments used by this method:
- `--method Chaos`
- `--npoints <int>`
- `--backend cpu|gpu|auto`
- `--resolution <HxW>`
- `--out <path>`

Example:

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl render --input Fractals.jl/data/Default.ifs --ifs-index 1 --method Chaos --npoints 200000 --backend cpu --resolution 1024x1024 --out media/cli_chaos.png
```

### `PointDeterministic`

Arguments used by this method:
- `--method PointDeterministic`
- `--iterations <int>`
- `--npoints <int>`
- `--backend cpu|gpu|auto`
- `--resolution <HxW>`
- `--out <path>`

Example:

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl render --input Fractals.jl/data/Default.ifs --ifs-index 1 --method PointDeterministic --iterations 3 --npoints 500 --backend cpu --resolution 1024x1024 --out media/cli_point_deterministic.png
```

### `ImageIterate`

Arguments used by this method:
- `--method ImageIterate`
- `--image-source polygon|chaos|point_deterministic|inverse|file`
- `--image-path <path>` (required when `--image-source file`)
- `--image-iterations <int>`
- `--polygon-limits-mode ifs|default`
- `--initial-polygon default|equilateral_triangle|line_arrow|line`
- `--backend cpu|gpu|auto`
- `--resolution <HxW>`
- `--out <path>`

Example:

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl render --input Fractals.jl/data/Default.ifs --ifs-index 1 --method ImageIterate --image-source polygon --image-iterations 3 --polygon-limits-mode ifs --initial-polygon equilateral_triangle --backend cpu --resolution 1024x1024 --out media/cli_image_iterate.png
```

### `Inverse`

Arguments used by this method:
- `--method Inverse`
- `--iterations <int>`
- `--backend cpu|gpu|auto`
- `--resolution <HxW>`
- `--out <path>`

Example:

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl render --input Fractals.jl/data/Default.ifs --ifs-index 1 --method Inverse --iterations 8 --backend cpu --resolution 1024x1024 --out media/cli_inverse.png
```

## Notes

- Methods accepted by CLI are the same as the API (`Chaos`, `Parallel`, `PointDeterministic`, `ImageIterate`, `Inverse`, plus symbol/string forms).
- `--backend cpu|gpu|auto` controls render backends (`make_image` and inverse rasterization paths).
  - `gpu` requires CUDA support and errors if unavailable.
  - `auto` silently falls back to CPU when GPU support is unavailable.
- `Parallel` currently maps to `Chaos` behavior and may be deprecated in a future release.
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
- For benchmark command:
  - `--backend cpu|gpu|auto` controls GPU-capable benchmark backends (`make_image`, `iterate_image`).
  - `--include-gpu-bench` adds explicit `make_image_gpu`, `iterate_image_gpu`, `rasterize_image_inversely_gpu_exact`, and `rasterize_image_inversely_gpu_preview` timing lines.
