from openfermion.ops import FermionOperator, MajoranaOperator
from openfermion.transforms import bravyi_kitaev, bravyi_kitaev_fast, get_sparse_operator
import numpy as np
from itertools import combinations_with_replacement
import openfermion as of
import sympy

# Does this make sense for non-multiples of 4?
N= 4 # Number of sites for a single SYK model
q = 4 # setting q = N is all to all connectivity
J = 1 # overall coupling  strength
mu = .01 # interaction strength
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)

J_var = 2**(q-1) * J**2 * factorial(q-1) / (q * N**(q-1))

# DONE: Figure out how to do the double copying
SYK_1_indices = list(combinations_with_replacement(range(N), q))
SYK_2_indices = list(combinations_with_replacement(range(N, 2*N), q))
# DONE: Generate couplings
np.random.seed(0)
couplings = np.random.normal(scale=np.sqrt(J_var), size=len(SYK_1_indices))
J_1_q = {i: c for i, c in zip(SYK_1_indices, couplings)}
J_2_q = {i: -c for i, c in zip(SYK_2_indices, couplings)}


def sum_ops(operators):
    # Wrapper for summing a list of majorana operators
    return sum(operators, MajoranaOperator((), 0))



# DONE: Iterate over couplings and produce corresponding operator
hamiltonian_1_terms = [MajoranaOperator(ind, J_1_q[ind]) for ind in SYK_1_indices]
total_hamiltonian_1 = sum_ops(hamiltonian_1_terms)
hamiltonian_2_terms = [MajoranaOperator(ind, J_2_q[ind]) for ind in SYK_2_indices]
total_hamiltonian_2 = sum_ops(hamiltonian_2_terms)

interaction_terms = [MajoranaOperator((i, i+N), 1j * mu) for i in range(N)]
interaction_hamiltonian = sum_ops(interaction_terms)

# BK_terms = [bravyi_kitaev(op) for op in hamiltonian_terms]
bk_hamiltonian_1 = bravyi_kitaev(total_hamiltonian_1)
bk_hamiltonian_2 = bravyi_kitaev(total_hamiltonian_2)
bk_interaction = bravyi_kitaev(interaction_hamiltonian)

total_ham = bk_hamiltonian_1 + bk_hamiltonian_2 + bk_interaction
matrix_ham = get_sparse_operator(total_ham) # todense() allows for ED

# TODO: See if can find the exact ground state of this new hamiltonian (Maybe in julia..)
def gs_energy(hamiltonian):
    from scipy.linalg import eigvalsh
    return eigvalsh(hamiltonian, eigvals=(0,0))

print(gs_energy(matrix_ham.todense()))

# TODO: Check out the hamiltonian as it's printed outermost
# DONE: Make routine to measure the energy of the hamiltonian in a circuit
# TODO: Compare the variational ground state
# This bit will probably be done variationally in the qubit basis, but not really sure that it matters too much
# TODO: Check out what alternating hamiltonians are good to try to fit this into QAOA
# print(bravyi_kitaev_fast(zero_dag_3))
