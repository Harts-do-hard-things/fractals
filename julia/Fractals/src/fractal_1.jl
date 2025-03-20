using Plots

# Define a struct for Fractal
struct Fractal
    S0::Vector{ComplexF64}
    func_list::Vector{Function}
    S::Vector{ComplexF64}
    plot_list::Vector{Vector{ComplexF64}}
end

# Constructor for Fractal
function Fractal(S0::Vector{ComplexF64}, func_list::Vector{Function})
    return Fractal(S0, func_list, S0, [S0])
end

# Function to iterate fractal
function iterate!(fractal::Fractal, i::Int)
    for _ in 1:i
        new_S = ComplexF64[]
        for func in fractal.func_list
            append!(new_S, func.(fractal.S))
        end
        fractal.S = new_S
        push!(fractal.plot_list, new_S)
    end
end

# Function to plot fractal
function plot_fractal(fractal::Fractal)
    plot()
    for s in fractal.plot_list
        scatter!(real.(s), imag.(s), markersize=1, legend=false)
    end
end

# Define transformations for Heighway Dragon
heighway_funcs = [
    z -> 0.5 * (1 + im) * z,
    z -> 1 - 0.5 * (1 - im) * z
]

# Initialize and generate fractal
heighway = Fractal([0 + 0im, 1 + 0im], heighway_funcs)
iterate!(heighway, 10)
plot_fractal(heighway)
