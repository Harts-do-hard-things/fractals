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
    out = render(IFS(eq; name = "main"), Chaos(npoints = npoints); outpath = outpath)
    println("Saved to $(out.outpath)")
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
