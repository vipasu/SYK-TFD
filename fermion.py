from openfermion.ops import FermionOperator, MajoranaOperator
from openfermion.transforms import bravyi_kitaev, bravyi_kitaev_fast
import numpy as np
from itertools import combinations_with_replacement

# Does this make sense for non-multiples of 4?
N= 4 # Number of sites for a single SYK model
q = 4 # setting q = N is all to all connectivity
J = 1 # overall coupling  strength
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)

J_var = 2**(q-1) * J**2 * factorial(q-1) / (q * N**(q-1))

majorana_indices = list(combinations_with_replacement(range(N), q))
# DONE: Generate couplings
np.random.seed(0)
J_q = {i: np.random.normal(scale=np.sqrt(J_var)) for i in majorana_indices}




# DONE: Iterate over couplings and produce corresponding operator
hamiltonian_terms = [MajoranaOperator(ind, J_q[ind]) for ind in majorana_indices]
total_hamiltonain = sum(hamiltonian_terms, MajoranaOperator((), 0))

# BK_terms = [bravyi_kitaev(op) for op in hamiltonian_terms]
bk_hamiltonian = bravyi_kitaev(total_hamiltonain)

# TODO: Check out the hamiltonian as it's printed outermost
# TODO: Figure out how to do the double copying
# TODO: See if can find the exact ground state of this new hamiltonian
# TODO: Compare the variational ground state
# This bit will probably be done variationally in the qubit basis, but not really sure that it matters too much
# TODO: Check out what alternating hamiltonians are good to try to fit this into QAOA
# print(bravyi_kitaev_fast(zero_dag_3))
