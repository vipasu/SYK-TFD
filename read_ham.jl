module read_ham
  using Yao, YaoExtensions
  using Plots
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
      lines = lines[2:end]
      # print(lines[1])
      construct_hamiltonian_from_terms(lines, N), SYK_energies
  end

  function read_annihilation(fname, N)
      lines = readlines(fname)
      construct_hamiltonian_from_terms(lines, N)
  end


  function test_annihilation(N, seed, circuit)
      annihilation_fname =  "SYK_annihilator_JW_$(N)_$(seed).txt"
      h = read_annihilation(annihilation_fname, N)
      return real.(expect(h, zero_state(N) => circuit))
  end


  function default_train(N, seed)
    ham_fname= "SYK_ham_JW_$(N)_$(seed).txt"
    ham, SYK_energies = read_hamiltonian(ham_fname, N)

    d = 50
    circuit = dispatch!(variational_circuit(N, d), :random)
    h = ham

    energies = []
    annihilators = []
    num_steps = 500
    etas = gradient_schedule(num_steps, .2, .02)
    energy = (real.(expect(h, zero_state(N)=>circuit)))
    push!(energies, energy)
    for i in 1:num_steps
        _, grad = expect'(h, zero_state(N) => circuit)
        dispatch!(-, circuit, etas[i]* grad)
        energy = (real.(expect(h, zero_state(N)=>circuit)))
        parity = test_annihilation(N, seed, circuit)
        if i % 10 == 0
            println("Step $i, energy = $energy")
        end

        push!(energies, energy)
        push!(annihilators, parity)
    end
    println(SYK_energies)

    println(test_annihilation(N, seed, circuit))
    energies, annihilators, SYK_energies
  end


  # plot(energies, show=true)

end
