function _render_transformations_image(
    ifs;
    width::Int = 1200,
    height::Int = 1200,
    show_base::Bool = false,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol} = initial_polygon(),
    polygon_limits_iterations::Integer = 1,
    color::Bool = true,
    axis::Bool = false,
    alpha::Bool = false,
)
    raster_ifs = _render_transformations_effective_ifs(
        ifs;
        initial_polygon_spec = initial_polygon_spec,
        polygon_limits_iterations = polygon_limits_iterations,
    )

    seed = _render_initial_polygon_seed_image(
        raster_ifs;
        width = width,
        height = height,
        initial_polygon_spec = initial_polygon_spec,
    )
    img = iterate_image(raster_ifs, seed; colors = color)
    if show_base
        _overlay_initial_polygon!(
            img,
            raster_ifs;
            width = width,
            height = height,
            initial_polygon_spec = initial_polygon_spec,
            color = color,
        )
    end
    if axis
        _overlay_axes!(img, raster_ifs; width = width, height = height)
    end
    return alpha ? _apply_alpha_mask(img) : img
end
