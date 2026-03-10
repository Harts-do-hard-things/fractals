#!/usr/bin/env julia

using Fractals

function rotation_eq(degrees::Real; clockwise::Bool)::Matrix{Float64}
    θ = Float64(degrees) * (pi / 180.0)
    c = cos(θ)
    s = sin(θ)
    # Canonical convention: positive angle is counterclockwise in model space.
    row = clockwise ? [c -s s c 0.0 0.0 1.0] : [c s -s c 0.0 0.0 1.0]
    return reshape(row, 1, 7)
end

cases = [
    (30.0, false, "rotation_30_ccw"),
    (30.0, true, "rotation_30_cw"),
    (45.0, false, "rotation_45_ccw"),
    (45.0, true, "rotation_45_cw"),
]

for (deg, clockwise, stem) in cases
    eq = rotation_eq(deg; clockwise=clockwise)
    ifs = IFS(eq; npoints=10, name=stem)
    svg_path = render_transformations_svg(ifs;
                                          outpath=joinpath("media", "$stem.svg"),
                                          width=640,
                                          height=640,
                                          initial_polygon=:line)
    png_path = render_transformations_png(ifs;
                                          outpath=joinpath("media", "$stem.png"),
                                          width=640,
                                          height=640,
                                          initial_polygon=:line,
                                          color=false)
    println(svg_path)
    println(png_path)
end
