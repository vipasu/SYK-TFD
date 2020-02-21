## Variational stuff
import cirq
import openfermioncirq as ofc
objective = ofc.HamiltonianObjective(total_ham)
class MyAnsatz(ofc.VariationalAnsatz):

    def params(self):
        """The parameters of the ansatz."""
        return [sympy.Symbol('theta_0')]

    def operations(self, qubits):
        """Produce the operations of the ansatz circuit."""
        q0, q1, q2, q3 = qubits
        yield cirq.H(q0), cirq.H(q1), cirq.H(q2)
        yield cirq.Rx(np.pi/2).on(q3)

        yield cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)
        # yield cirq.Rz(sympy.Symbol('theta_0')).on(q3)
        yield cirq.CNOT(q2, q3), cirq.CNOT(q1, q2), cirq.CNOT(q0, q1)

        yield cirq.H(q0), cirq.H(q1), cirq.H(q2)
        yield cirq.Rx(np.pi/2).on(q3)

    def _generate_qubits(self):
        """Produce qubits that can be used by the ansatz circuit."""
        return cirq.LineQubit.range(4)

ansatz = MyAnsatz()
q0, q1, q2, q3 = ansatz.qubits
prep_circuit = cirq.Circuit.from_ops(cirq.X(q2), cirq.X(q0))
prep_circuit = cirq.Circuit()


study = ofc.VariationalStudy(
    name='my_test_study',
    ansatz=ansatz,
    objective=objective,
    preparation_circuit=prep_circuit)
print(study.value_of(np.array([0.0])))

for p in np.linspace(-np.pi, np.pi, 101):
    print(study.value_of(np.array([p])))
