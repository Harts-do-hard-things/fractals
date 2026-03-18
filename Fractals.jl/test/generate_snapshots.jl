using Fractals
using FileIO

const SNAPSHOT_DIR = joinpath(@__DIR__, "snapshots")

const SMALL_EQ = [
    0.5  0.0  0.0  0.5  0.0  0.0  0.6;
   -0.5  0.0  0.0 -0.5  1.0  0.0  0.4
]

mkpath(SNAPSHOT_DIR)

function _snapshot_path(name::AbstractString)
    return joinpath(SNAPSHOT_DIR, name * ".png")
end

function _write_snapshot(name::AbstractString, image)
    path = _snapshot_path(name)
    save(path, image)
    println(path)
end

transform_ifs = IFS(SMALL_EQ; npoints=50)
transform_img = Fractals._render_transformations_image(transform_ifs;
                                                       width=96,
                                                       height=96,
                                                       initial_polygon=:line_arrow,
                                                       color=true,
                                                       axis=true)
_write_snapshot("transformations-line-arrow-color-axis", transform_img)

inverse = render(SMALL_EQ;
                 method=Inverse,
                 iterations=2,
                 show_divergence_scale=false,
                 backend=:cpu,
                 resolution=(32, 32),
                 outpath=joinpath("media", "snapshot_inverse.png"))
_write_snapshot("inverse-mask-small", inverse.image)

image_iter = render(SMALL_EQ;
                    method=ImageIterate,
                    image_source=:polygon,
                    image_iterations=2,
                    backend=:cpu,
                    resolution=(48, 48),
                    outpath=joinpath("media", "snapshot_image_iterate.png"))
_write_snapshot("image-iterate-polygon-small", image_iter.image)
