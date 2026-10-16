# --------------------------------
# Example Systems
# --------------------------------

"Equation matrix for the two-map Heighway dragon (6 columns; weights from determinants)."
const HEIGHWAY_DRAGON = [
    0.5 -0.5 0.5 0.5 0.0 0.0
    -0.5 -0.5 0.5 -0.5 1.0 0.0
]

"Equation matrix for a four-map Eisenstein-integer fractal with equal probabilities."
const EISENSTEIN = [
    -0.5 0.0 0.0 -0.5 0.0 0.0 0.25
    -0.5 0.0 0.0 -0.5 -0.5 0.0 0.25
    -0.5 0.0 0.0 -0.5 0.25 -0.433 0.25
    -0.5 0.0 0.0 -0.5 0.25 0.433 0.25
]

# --------------------------------
# Main
# --------------------------------

function main(; eq = EISENSTEIN, npoints = 1_000_000, outpath = nothing)

    println("Building IFS...")
    ifs = IFS(eq; npoints = npoints)

    println("Running chaos game...")
    iterate!(ifs)

    println("Rasterizing...")
    img = make_image(ifs)

    params = (; method = :chaos, ifs_name = ifs.name, npoints, resolution = RESOLUTION)
    final_outpath =
        _resolve_output_path(outpath, _output_name("main", params[(:method, :npoints)], "png"))
    println("Saving to $final_outpath")
    _save_image(final_outpath, img, ifs.limits; params = params)

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
