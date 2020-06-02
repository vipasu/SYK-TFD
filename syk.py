from openfermion.ops import FermionOperator, MajoranaOperator
from openfermion.transforms import bravyi_kitaev, bravyi_kitaev_fast, get_sparse_operator
import numpy as np
from itertools import combinations
import openfermion as of
import sympy
import cirq
import argparse
from scipy.linalg import eigh


parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=8, help="Number of fermions")
parser.add_argument("--seed", dest="seed", default=0, type=int, help="Random seed")
args = parser.parse_args()
# print(args)




# Does this make sense for non-multiples of 4?
N= args.N # Number of sites for a single SYK model
q = 4 # setting q = N is all to all connectivity
J = 1 # overall coupling  strength
mu = .01 # interaction strength
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)

J_var = 2**(q-1) * J**2 * factorial(q-1) / (q * N**(q-1))

# DONE: Figure out how to do the double copying
SYK_1_indices = list(combinations(range(N), q))
SYK_2_indices = list(combinations(range(N, 2*N), q))
# DONE: Generate couplings
np.random.seed(args.seed)
couplings = np.random.normal(scale=np.sqrt(J_var), size=len(SYK_1_indices))
J_1_q = {i: c for i, c in zip(SYK_1_indices, couplings)}
J_2_q = {i: -c for i, c in zip(SYK_2_indices, couplings)}


def sum_ops(operators):
    # Wrapper for summing a list of majorana operators
    return sum(operators, MajoranaOperator((), 0))



# DONE: Iterate over couplings and produce corresponding operator
# Insert factors of i here if there are terms that don't have 4 terms in them
hamiltonian_1_terms = [MajoranaOperator(ind, J_1_q[ind]) for ind in SYK_1_indices]
total_hamiltonian_1 = sum_ops(hamiltonian_1_terms)
hamiltonian_2_terms = [MajoranaOperator(ind, J_2_q[ind]) for ind in SYK_2_indices]
total_hamiltonian_2 = sum_ops(hamiltonian_2_terms)

interaction_terms = [MajoranaOperator((i, i+N), 1j * mu) for i in range(N)]
interaction_hamiltonian = sum_ops(interaction_terms)

# BK_terms = [bravyi_kitaev(op) for op in hamiltonian_terms]
bk_hamiltonian_1 = bravyi_kitaev(total_hamiltonian_1, N)
bk_hamiltonian_2 = bravyi_kitaev(total_hamiltonian_2, N)
bk_interaction = bravyi_kitaev(interaction_hamiltonian, N)

total_ham = bk_hamiltonian_1 + bk_hamiltonian_2 + bk_interaction
matrix_ham = get_sparse_operator(total_ham) # todense() allows for ED

# DONE: See if can find the exact ground state of this new hamiltonian (Maybe in julia..)
def gs_energy(hamiltonian):
    from scipy.linalg import eigvalsh
    return eigvalsh(hamiltonian, eigvals=(0,0))

# print(gs_energy(matrix_ham.todense()))

# DONE: Make routine to measure the energy of the hamiltonian in a circuit
# TODO: Compare the variational ground state
# This bit will probably be done variationally in the qubit basis, but not really sure that it matters too much


def construct_pauli_string(ham, key):
    gate_dict = {'X': cirq.X,
                 'Y': cirq.Y,
                 'Z': cirq.Z}
    def q_helper(idx):
        return cirq.LineQubit(idx)

    def list_of_terms(key):
        return [gate_dict[label](q_helper(idx)) for (idx, label) in key]


    return cirq.PauliString(ham.terms[key], list_of_terms(key))
qubits = cirq.LineQubit.range(N)
prep_circuit = cirq.Circuit(*([cirq.Z(qubits[i]) for i in range(N)] +
                            [cirq.X(qubits[1])]))
psi = cirq.final_wavefunction(prep_circuit)
all_strings = list(total_ham.terms)
test_string = construct_pauli_string(total_ham, all_strings[0])

e, v = eigh(matrix_ham.todense())
print("E0, E1 = {},{}".format(e[0], e[1]))
for k, v in total_ham.terms.items():
    print("{} => {}".format(np.real(v), k))
# Debugging BK on individual terms
# for t in hamiltonian_1_terms:
#     print(t, bravyi_kitaev(t))
