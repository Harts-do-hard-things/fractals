# FractalsGUI

Desktop GUI package for editing IFS transformation matrices and previewing transformation SVG output using `Fractals`.

## Launch

```bash
julia --startup-file=no --project=FractalsGUI.jl FractalsGUI.jl/bin/gui.jl
```

## Run tests

```bash
julia --startup-file=no --project=FractalsGUI.jl -e "using Pkg; Pkg.test()"
```

This test suite remains separate from `Fractals.jl/test/runtests.jl`, and the non-interactive suite is also exercised by repository CI on Ubuntu.
