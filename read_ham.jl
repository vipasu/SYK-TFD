using Yao, YaoExtensions
using Plots

function extract_pauli_term(line)
    first_space_ind = findfirst(isequal(' '), line)
    coefficient = parse(Float64, line[1:first_space_ind-1])
    second_space_ind = findnext(isequal(' '), line, first_space_ind + 1)
    terms = eval(Meta.parse(line[second_space_ind + 1:end]))
    return coefficient, terms
end



nbit = 12
lines = readlines("SYK_$(nbit)_ham.txt")
SYK_energies = lines[1]
lines = lines[2:end]
# print(lines[1])
sx = i->put(nbit, i=>X)
sy = i->put(nbit, i=>Y)
sz = i->put(nbit, i=>Z)
ham = map(lines) do line
    val, pauli_terms = extract_pauli_term(line)
    pauli_string =  (map(pauli_terms) do item
        q_ind, pauli_choice = item
        q_ind = q_ind + 1 # conversion from python
        if pauli_choice == 'X'
            return sx(q_ind)
        elseif pauli_choice == 'Y'
            return sy(q_ind)
        elseif pauli_choice == 'Z'
            return sz(q_ind)
        else
            return "whips"
        end
    end |> prod)
    val * pauli_string
end |> sum

n, d = nbit, 100
circuit = dispatch!(variational_circuit(n, d), :random)
h = ham

energies = []
for i in 1:200
    _, grad = expect'(h, zero_state(n) => circuit)
    dispatch!(-, circuit, 1e-2* grad)
    energy = (real.(expect(h, zero_state(n)=>circuit)))
    println("Step $i, energy = $energy")
    push!(energies, energy)
end
println(SYK_energies)

# plot(energies, show=true)
