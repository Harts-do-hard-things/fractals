# FractalsGUI

Desktop GUI package for editing IFS transformation matrices and previewing transformation SVG output using `Fractals`.

## Launch

```bash
julia --startup-file=no --project=FractalsGUI.jl FractalsGUI.jl/bin/gui.jl
```

## Run tests (local only)

```bash
julia --startup-file=no --project=FractalsGUI.jl -e "using Pkg; Pkg.test()"
```

This test suite is intentionally separate from `Fractals/test/runtests.jl` and is not wired into repository CI.
