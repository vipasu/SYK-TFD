module read_ham
    using Yao, YaoExtensions
    using Plots
    using Printf
    export extract_pauli_term, construct_hamiltonian_from_terms, gradient_schedule, read_hamiltonian, read_annihilation, test_annihilation
    export default_train

    function extract_pauli_term(line)
        first_space_ind = findfirst(isequal(' '), line)
        coefficient = parse(Float64, line[1:first_space_ind-1])
        second_space_ind = findnext(isequal(' '), line, first_space_ind + 1)
        terms = eval(Meta.parse(line[second_space_ind + 1:end]))
        return coefficient, terms
    end

    function construct_hamiltonian_from_terms(lines, N)
        sx = i->put(N, i=>X)
        sy = i->put(N, i=>Y)
        sz = i->put(N, i=>Z)
        ham = map(lines) do line
            coef, pauli_terms = extract_pauli_term(line)
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
            coef * pauli_string
        end |> sum
    end


    function gradient_schedule(num_iters, eta_1, eta_0)
        map(x -> eta_1/x + eta_0, 1:num_iters)
    end



    function read_hamiltonian(fname, N)
        lines = readlines(fname)
        SYK_energies = map(x -> parse(Float64, x), split(lines[1],","))
        left_lines = lines[2:Int((end-N+1)//2)]
        right_lines = lines[Int((end-N+3)//2):end-N]
        int_lines = lines[end-N+1:end]
        # print(lines[1])
        h_L = construct_hamiltonian_from_terms(left_lines, N)
        h_R = construct_hamiltonian_from_terms(right_lines, N)
        h_int = construct_hamiltonian_from_terms(int_lines, N)
        h_L, h_R, h_int, SYK_energies
    end

    function read_annihilation(fname, N)
        lines = readlines(fname)
        construct_hamiltonian_from_terms(lines, N)
    end


    function test_annihilation(N, seed, circuit, mu)
        annihilation_fname =  "data/SYK_annihilator_$(N)_$(seed)_$(@sprintf("%.2f", mu)).txt"
        h = read_annihilation(annihilation_fname, N)
        return real.(expect(h, zero_state(N) => circuit))
    end


    function default_train(N, seed, mu=.01, d=50, num_steps=500)
        ham_fname= "data/SYK_ham_$(N)_$(seed)_$(@sprintf("%.2f", mu)).txt"
        ham, SYK_energies = read_hamiltonian(ham_fname, N)

        circuit = dispatch!(variational_circuit(N, d), :zero)
        h = ham

        energies = []
        annihilators = []
        etas = gradient_schedule(num_steps, .4, .04)
        energy = (real.(expect(h, uniform_state(N)=>circuit)))
        push!(energies, energy)
        for i in 1:num_steps
            _, grad = expect'(h, uniform_state(N) => circuit)
            dispatch!(-, circuit, etas[i]* grad)
            energy = (real.(expect(h, uniform_state(N)=>circuit)))
            parity = test_annihilation(N, seed, circuit, mu)
            if i % 10 == 0
                println("Step $i, energy = $energy")
            end

            push!(energies, energy)
            push!(annihilators, parity)
        end
        println(SYK_energies)

        println(test_annihilation(N, seed, circuit, mu))
        energies, annihilators, SYK_energies
    end


end
