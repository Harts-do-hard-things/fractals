# Build the Fractals.jl documentation site with Documenter.jl.
#
#   julia --startup-file=no --project=docs -e "using Pkg; Pkg.instantiate()"
#   julia --startup-file=no --project=docs docs/make.jl
#
# Output goes to docs/build/. The build fails on undocumented exports, broken @ref links, and
# failing doctests. Nothing is deployed; add `deploydocs(...)` to publish to gh-pages.

using Documenter
using Fractals
using IFSParse

# Doctests run with Fractals (and StaticArrays for SVector) loaded, and write any render
# output to a scratch directory rather than the project's plots/.
DocMeta.setdocmeta!(
    Fractals,
    :DocTestSetup,
    :(using Fractals, StaticArrays; ENV["FRACTALS_OUTPUT_ROOT"] = mktempdir());
    recursive = true,
)

makedocs(;
    sitename = "Fractals.jl",
    modules = [Fractals, IFSParse],
    authors = "cub525",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = nothing,
        size_threshold_ignore = ["api.md"],
        inventory_version = string(pkgversion(Fractals)),
    ),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "manual/render_methods.md",
            "manual/quick_recipes.md",
            "manual/animation.md",
            "manual/output.md",
            "manual/data_layout.md",
            "manual/ifs_format.md",
            "manual/benchmarks.md",
            "manual/troubleshooting.md",
            "manual/development.md",
        ],
        "API reference" => "api.md",
    ],
)
