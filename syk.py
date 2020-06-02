from openfermion.ops import FermionOperator, MajoranaOperator
from openfermion.transforms import bravyi_kitaev, bravyi_kitaev_fast, get_sparse_operator
import numpy as np
from itertools import combinations
import openfermion as of
import sympy
import cirq
import argparse
from scipy.linalg import eigh


def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)

def get_couplings(N, var, L_inds, R_inds, seed, q):
    """Returns dictionaries of hamiltonian terms and their coefficients"""
    np.random.seed(args.seed)
    couplings = np.random.normal(scale=np.sqrt(var), size=len(L_inds))
    phase = (-1)**(q/2)
    J_L = {i: c for i, c in zip(L_inds, couplings)}
    J_R = {i: phase * c for i, c in zip(R_inds, couplings)}
    return J_L, J_R

def convert_H_majorana_to_qubit(inds, J_dict, N):
    """Convert SYK hamiltonian (dictionary) from majorana terms to Pauli terms"""
    ham_terms = [MajoranaOperator(ind, J_dict[ind]) for ind in inds]
    ham_sum = sum_ops(ham_terms)
    return bravyi_kitaev(ham_sum, N)



def q_helper(idx):
    return cirq.LineQubit(idx)


def construct_pauli_string(ham, key):
    gate_dict = {'X': cirq.X,
                'Y': cirq.Y,
                'Z': cirq.Z}

    def list_of_terms(key):
        return [gate_dict[label](q_helper(idx)) for (idx, label) in key]


    return cirq.PauliString(ham.terms[key], list_of_terms(key))


def sum_ops(operators):
    # Wrapper for summing a list of majorana operators
    return sum(operators, MajoranaOperator((), 0))


def gs_energy(hamiltonian):
    from scipy.linalg import eigvalsh
    return eigvalsh(hamiltonian, eigvals=(0,0))


def main(N, seed):
    # Does this make sense for non-multiples of 4?
    #N Number of sites for a single SYK model
    q = 4 # setting q = N is all to all connectivity
    J = 1 # overall coupling  strength
    mu = .01 # interaction strength

    J_var = 2**(q-1) * J**2 * factorial(q-1) / (q * N**(q-1))

    # DONE: Figure out how to do the double copying
    SYK_L_indices = list(combinations(range(N), q))
    SYK_R_indices = list(combinations(range(N, 2*N), q))
    interaction_indices = [(i, i+N) for i in range(N)]
    # DONE: Generate couplings
    J_L, J_R = get_couplings(N, J_var, SYK_L_indices, SYK_R_indices, seed, q)
    interaction_strength = {ind: 1j * mu for ind in interaction_indices}

    H_L = convert_H_majorana_to_qubit(SYK_L_indices, J_L, N)
    H_R = convert_H_majorana_to_qubit(SYK_R_indices, J_R, N)
    H_int = convert_H_majorana_to_qubit(interaction_indices, interaction_strength, N)

    total_ham = H_L + H_R + H_int
    annihilation_ham = H_L - H_R
    matrix_ham = get_sparse_operator(total_ham) # todense() allows for ED


    all_strings = list(total_ham.terms)
    test_string = construct_pauli_string(total_ham, all_strings[0])

    # Diagonalize qubit hamiltonian to compare the spectrum of variational energy
    e, v = eigh(matrix_ham.todense())
    n_spec = 4


    # Write out qubit hamiltonian to file
    fname = "SYK_ham_{}_{}.txt".format(N, seed)

    with open(fname, 'w') as f:
        e_string = ",".join(map(str, e[:n_spec]))
        f.write(e_string + '\n')
        for k, v in total_ham.terms.items():
            f.write("{} => {}\n".format(np.real(v), k))

    # Write out annihilator to file
    fname = "SYK_annihilator_{}_{}.txt".format(N, seed)
    with open(fname, 'w') as f:
        for k, v in annihilation_ham.terms.items():
            f.write("{} => {}\n".format(np.real(v), k))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=8, help="Number of fermions")
    parser.add_argument("--seed", dest="seed", default=0, type=int, help="Random seed")
    args = parser.parse_args()
    # print(args)
    main(args.N, args.seed)
