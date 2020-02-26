import pathos.pools as pp
from functools import partial
import numpy as np
from scipy import linalg
import cirq
import dill as pickle

class PQC():
    def __init__(self, parameters=None, n_qubits=8, n_layers=6, eta=.05, sample_grad=-1, *args, **kwargs):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.eta = eta
        # number of entries to sample for the gradient; -1 for all
        self.sample_grad = sample_grad

        if not parameters:
            self.parameters = np.random.randn(n_qubits * n_layers).reshape(n_layers, n_qubits)
        else:
            self.parameters = parameters

    def evaluate(self, parameters, state):
        """Define backend specific circuit"""
        raise NotImplementedError

    def gradient_helper(self, parameters, state, idx):
        """Evaluate gradient by parameter shift rule"""
        delta = np.pi/4
        parameters[idx] += delta
        positive_phase = self.evaluate(parameters, state)
        parameters[idx] -= 2 * delta
        negative_phase = self.evaluate(parameters, state)
        return positive_phase - negative_phase, idx


    def gradient_sub_sample(self, state):
        grad = np.zeros_like(self.parameters)
        param_indices = np.ndindex(self.parameters.shape)
        indices = []
        indices_subset = []
        for index in param_indices:
            indices.append(index)
        sample = np.unique(np.random.choice(len(self.parameters), self.sample_grad))
        for i in sample:
            indices_subset.append(indices[i])

        local_gradient_helper = partial(self.gradient_helper, self.parameters, state)
        pool = pp.ProcessPool()
        results = pool.map(local_gradient_helper, indices_subset)
        for (component, idx) in results:
            grad[idx] += component
        return grad


    def gradient(self, state):
        """Parallel implementation of circuit gradient

        Note that thetas will have a factor of 2 which needs to be accounted for."""
        grad = np.zeros_like(self.parameters)
        param_indices = np.ndindex(self.parameters.shape)

        local_gradient_helper = partial(self.gradient_helper, self.parameters, state)
        pool = pp.ProcessPool()
        results = pool.map(local_gradient_helper, param_indices)
        for (component, idx) in results:
            grad[idx] += component
        return grad

    def update(self, states, labels):
        """Update parameters based on batch gradient"""
        grad = np.zeros_like(self.parameters)
        for x, y in zip(states, labels):
            if self.sample_grad < 0:
                grad += (1-y * self.evaluate(self.parameters, x))* y*self.gradient(x)
            else:
                grad += (1-y * self.evaluate(self.parameters, x))* y*self.gradient_sub_sample(x)
        grad_norm_sq = 1. #np.sum(grad**2)
        self.parameters += self.eta * grad/grad_norm_sq

    def accuracy(self, states, labels):
        """Evaluate accuracy on given dataset"""

        local_eval = partial(self.evaluate, self.parameters)
        pool = pp.ProcessPool()
        predictions = pool.map(local_eval, states)
        # TODO: Put in mean label here rather than exact sign
        correct = [np.sign(pred) == label for pred, label in zip(predictions, labels)]

        print("mean margin loss:", np.mean([1 - pred * label for pred, label in zip(predictions, labels)]))
        return np.mean(correct)

    def pickle_self(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def save_parameters(self, fname):
        np.save(fname, self.parameters)

    def load_parameters_from_file(self, fname):
        self.parameters = np.load(fname)


class PQC_cirq(PQC):
    def __init__(self, layers, params_per_layer, *args, **kwargs):
        print(layers)
        print(args)
        print(kwargs)
        super(PQC_cirq, self).__init__(*args, **kwargs)
        self.simulator = cirq.Simulator()
        # self.construct_FN_circuit()
        self.layers = layers
        self.params_per_layer = params_per_layer
        self.parameters = np.random.randn(np.sum(params_per_layer))
        self.train_accuracies = []
        self.test_accuracies = []

    def exp_ZX(self, theta, q1, q2):
        instructions = []
        instructions.append(cirq.H(q2))
        instructions.append(cirq.CNOT(q1, q2))
        instructions.append(cirq.Rz(2*theta)(q2))
        instructions.append(cirq.CNOT(q1, q2))
        instructions.append(cirq.H(q2))
        return instructions

    def exp_XX(self, theta, q1, q2):
        instructions = []
        instructions.append(cirq.H(q1))
        instructions.extend(self.exp_ZX(theta, q1, q2))
        instructions.append(cirq.H(q1))
        return instructions

    def construct_random_circuit(self):
        # Initialize parameters of the circuit on the outside
        single_qubit_gates = np.random.choice([cirq.Rx, cirq.Ry, cirq.Rz], len(self.parameters))
        A = list(range(self.n_qubits//2))
        B = list(range(self.n_qubits//2, self.n_qubits))
        two_qubit_pairs_A = np.random.choice(A, self.n_layers)
        two_qubit_pairs_B = np.random.choice(B, self.n_layers)
        xx_pairs = list(zip(two_qubit_pairs_A, two_qubit_pairs_B))

        def random_circuit(parameters, program):
            qubits = [cirq.LineQubit(i) for i in range(self.n_qubits)]
            for layer in range(self.n_layers):
                # single qubit gates
                for i, q in enumerate(qubits):
                    idx = self.n_qubits * layer + i
                    # Apply a random 1 qubit gate with parameter theta on qubit q
                    program.append(single_qubit_gates[idx](parameters[idx])(q))

                # two qubit gate in between
                program.append(cirq.XX(qubits[xx_pairs[layer][0]], qubits[xx_pairs[layer][1]]))
            # Measure all qubits
            for q in qubits:
                program.append(cirq.measure(q))
            return program
        self.circuit = random_circuit


    def construct_FN_circuit(self):
        def fn_circuit(parameters, program):
            qubits = [cirq.LineQubit(i) for i in range(self.n_qubits)]
            last = qubits[-1]

            num_layers = self.parameters.shape[0]//2
            for layer in range(num_layers):
                zx_weights = self.parameters[2 * layer, :]
                xx_weights = self.parameters[2 * layer + 1, :]

                # Apply layer of ZX gates
                for i in range(self.n_qubits - 1):
                    program.append(self.exp_ZX(zx_weights[i], qubits[i], last))

                # Apply layer of XX gates
                for i in range(self.n_qubits - 1):
                    program.append(self.exp_XX(xx_weights[i], qubits[i], last))

            # program.append(cirq.measure(last))
            for q in qubits:
                program.append(cirq.measure(q))
            return program
        self.circuit = fn_circuit

    def circuit(self, layers, parameters, state):
        "assumes that layers are functions which accept parameters (concatenated already?)"
        subprograms = []
        theta_start = 0
        for layer, n_param in zip(layers, self.params_per_layer):
            subprograms.append(layer(parameters[theta_start:theta_start + n_param]))
            theta_start += n_param
        return sum(subprograms, state)

    def evaluate(self, parameters, state):
        program = state.copy()
        qubits = [cirq.LineQubit(i) for i in range(self.n_qubits)]
        last = qubits[-1]

        program = self.circuit(self.layers, parameters, program)
        program += cirq.Circuit.from_ops([cirq.measure(q) for q in qubits])
        results = self.simulator.run(program, repetitions=1000)
        # shift from 0 and 1 to -1 and 1
        # expectation = np.mean(results.measurements[f'{self.n_qubits - 1}'])
        expectations = [np.mean(results.measurements[f'{i}']) for i in range(self.n_qubits)]
        expectation = 2 * (np.mean(expectations) - .5)
        return expectation
