# --------------------------------
# Example Systems
# --------------------------------

const HEIGHWAY_DRAGON = [
     0.5  -0.5   0.5   0.5   0.0   0.0;
    -0.5  -0.5   0.5  -0.5   1.0   0.0
]

const EISENSTEIN = [
-0.5 0.0 0.0 -0.5  0.0   0.0  0.25;
-0.5 0.0 0.0 -0.5 -0.5   0.0  0.25;
-0.5 0.0 0.0 -0.5  0.25 -0.433 0.25;
-0.5 0.0 0.0 -0.5  0.25  0.433 0.25;
]

# --------------------------------
# Main
# --------------------------------

function main(; eq=EISENSTEIN,
               npoints=1_000_000,
               outpath="media/output.png")

    println("Building IFS...")
    ifs = IFS(eq; npoints=npoints)

    println("Running chaos game...")
    iterate!(ifs)

    println("Rasterizing...")
    img = make_image(ifs)

    final_outpath = _normalize_media_outpath(outpath)
    println("Saving to $final_outpath")
    save(final_outpath, img)

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
