# Render a small parameter sweep of the bundled example systems into plots/.
#
#   julia --startup-file=no --project=. scripts/render_examples.jl
#
# Each file is auto-named with DrWatson's savename from its parameters, re-running keeps the
# previous images as `_#1` backups, and every PNG carries provenance (see read_provenance).

using DrWatson
@quickactivate "fractals"

using Fractals

systems = Dict("Heighway Dragon" => HEIGHWAY_DRAGON, "Eisenstein" => EISENSTEIN)
sweep = dict_list(Dict(:name => collect(keys(systems)),
                       :npoints => [50_000, 200_000],
                       :color => [false, true]))

for params in sweep
    ifs = IFS(systems[params[:name]]; name=params[:name])
    out = render(ifs; method=Chaos, npoints=params[:npoints], color=params[:color], resolution=(800, 800))
    println(relpath(out.outpath, projectdir()))
end
