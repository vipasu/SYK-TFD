using Plots
using LaTeXStrings
using ArgParse
include("./read_ham.jl")
using .read_ham


s = ArgParseSettings()
@add_arg_table! s begin
    "-N"
    help = "Number of fermions in a single SYK"
    arg_type = Int
    default = 8
    "--num_steps"
    help = "Number of gradient iterations"
    arg_type = Int
    default = 2000
    "--num_layers"
    help = "Depth of PQC"
    arg_type = Int
    default = 20
    "--mu"
    help = "Interaction strength"
    arg_type = Float64
    default = .01
    "--plot"
    help = "Plot the training curves in Julia"
    arg_type = Bool
    default = false
    "--save"
    help = "Save the training data"
    arg_type = Bool
    default = true
    "--desc"
    help = "Description to add to the fname"
    arg_type = String
    default = ""
end

args = parse_args(s)

plot_jl = args["plot"]
N = args["N"]
num_steps = args["num_steps"]
num_layers = args["num_layers"]
mu = args["mu"]
save_flag = args["save"]
desc = args["desc"]

e1, a1, SYK1 = default_train(N, 0, mu, num_layers, num_steps)
e2, a2, SYK2 = default_train(N, 1, mu, num_layers, num_steps)
e3, a3, SYK3 = default_train(N, 2, mu, num_layers, num_steps)
e4, a4, SYK4 = default_train(N, 3, mu, num_layers, num_steps)

if plot_jl

    default(size=(800, 800))
    p = plot(title="Variational TFD", grid = false, showaxis = false, showframe=false, bottom_margin = -35Plots.px)
    f1 = plot(e1, title="Model a", label=L"$E(\theta)$", ylabel="Energy", legendfonthalign=:center, xlim=(0, num_steps))
    hline!([SYK1[1] SYK1[2] ], label=[L"E_0" L"E_1"], linestyle=:dash)
    f2 = plot(e2, title="Model b", label="", xlim=(0, num_steps))
    hline!([SYK2[1] SYK2[2] ],  label="",linestyle=:dash)
    f3 = plot(e3, title="Model c", label="", xlabel="Iterations", ylabel="Energy", xlim=(0, num_steps))
    hline!([SYK3[1] SYK3[2] ],  label="",linestyle=:dash)
    f4 = plot(e4, title="Model d", label="", xlabel="Iterations", xlim=(0, num_steps))
    hline!([SYK4[1] SYK4[2] ],  label="",linestyle=:dash)

    display(plot(p, f1, f2, f3, f4, layout=@layout([A{.0001h}; [B C]; [D E]])))

    # q = plot(title=L"H_L - H_R", grid = false, showaxis = false, showframe=false, bottom_margin = -35Plots.px)
    # g1 = plot(a1, title="Model a", label="", ylabel="Energy", legendfonthalign=:center, xlim=(0, 500), ylim=(-.1, .1))
    # hline!([0],  label="",linestyle=:dash)
    # g2 = plot(a2, title="Model b", label="", xlim=(0, 500), ylim=(-.1, .1))
    # hline!([0],  label="",linestyle=:dash)
    # g3 = plot(a3, title="Model c", label="", xlabel="Iterations", ylabel="Energy", xlim=(0, 500), ylim=(-.1, .1))
    # hline!([0],  label="",linestyle=:dash)
    # g4 = plot(a4, title="Model d", label="", xlabel="Iterations", xlim=(0, 500), ylim=(-.1, .1))
    # hline!([0],  label="",linestyle=:dash)
    # display(plot(q, g1, g2, g3, g4, layout=@layout([A{.0001h}; [B C]; [D E]])))

end

if save_flag
    using DelimitedFiles
    if isempty(desc)
        energy_fname = "data/energies_$(num_layers)_$(num_steps).txt"
        annihilator_fname = "data/annihilators_$(num_layers)_$(num_steps).txt"
    else
        energy_fname = "data/energies_$(num_layers)_$(num_steps)_$(desc).txt"
        annihilator_fname = "data/annihilators_$(num_layers)_$(num_steps)_$(desc).txt"
    end

    open(energy_fname, "w") do io
        writedlm(io, [e1'; e2'; e3'; e4'], ", ")
    end

    open(annihilator_fname, "w") do io
        writedlm(io, [a1'; a2'; a3'; a4'], ", ")
    end

end
