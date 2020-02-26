import numpy as np
import prep
from tqdm import tqdm
from PQC import *
import sys
import argparse
import itertools


def bellCreate(q1, q2):
    """Assumes two qubits are in the 00 state to start and creates the singlet state (ket(+-) - ket(-+))"""
    instructions = []
    instructions.append(cirq.X(q1))
    instructions.append(cirq.H(q1))
    instructions.append(cirq.CNOT(q1, q2))
    return instructions

def prepare_inf_tfd(n_qubits):
    A_size = n_qubits //2
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    instructions = []
    for a, b in zip(qubits[:A_size], qubits[A_size:]):
        instructions.extend(bellCreate(a, b))

    return cirq.Circuit.from_ops(instructions)

def apply_ising_xx(n_qubits, parameters):
    """Apply trotterized H_A for XX component"""
    A_size = n_qubits //2
    qubits = [cirq.LineQubit(i) for i in range(A_size)]
    instructions = []
    for i, p in enumerate(parameters):
        instructions.append(cirq.XXPowGate(p).on(qubits[i], qubits[(i+1) % A_size]))
    return cirq.Circuit.from_ops(instructions)


def apply_ising_z(n_qubits):
    """Apply trotterized H_A for Z component"""
    A_size = n_qubits //2
    qubits = [cirq.LineQubit(i) for i in range(A_size)]
    instructions = []
    for i, p in enumerate(parameters):
        instructions.append(cirq.Rz(p).on(qubits[i]))
    return cirq.Circuit.from_ops(instructions)

def apply_mixing_yy(n_qubits, parameters):
    A_size = n_qubits //2
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    instructions = []
    for a, b, theta in zip(qubits[:A_size], qubits[A_size:], parameters):
        instructions.append(cirq.YYPowGate(theta).on(a, b))
    return cirq.Circuit.from_ops(instructions)

def apply_mixing_xx(n_qubits, parameters):
    A_size = n_qubits //2
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    instructions = []
    for a, b, theta in zip(qubits[:A_size], qubits[A_size:], parameters):
        instructions.append(cirq.XXPowGate(theta).on(a, b))
    return cirq.Circuit.from_ops(instructions)

def apply_mixing_zz(n_qubits, parameters):
    A_size = n_qubits //2
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    instructions = []
    for a, b, theta in zip(qubits[:A_size], qubits[A_size:], parameters):
        instructions.append(cirq.ZZPowGate(theta).on(a, b))
    return cirq.Circuit.from_ops(instructions)

def RZ_layer(parameters):
    qubits = [cirq.LineQubit(i) for i in range(8)]
    operations = []
    for q, p in zip(qubits, parameters):
        operations.append(cirq.Rz(p).on(q))
    return cirq.Circuit.from_ops(operations)

def RX_layer(parameters):
    qubits = [cirq.LineQubit(i) for i in range(8)]
    operations = []
    for q, p in zip(qubits, parameters):
        operations.append(cirq.Rx(p).on(q))
    return cirq.Circuit.from_ops(operations)

def CRx(q1, q2, theta):
    """Helper function for controlled RX gate"""
    return cirq.ControlledGate(cirq.Rx(theta)).on(q1, q2)

def CRx_layer(parameters):
    """Expects 16 parameters"""
    n_qubits = 8
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    A = qubits[:n_qubits//2]
    B = qubits[n_qubits//2:]
    operations = []
    for i, (a, b) in enumerate(itertools.product(A,B)):
        operations.append(CRx(a, b, parameters[i]))
    return cirq.Circuit.from_ops(operations)

def main(args):
    # x_train, y_train, x_test, y_test = prep.entangled_dataset()
    layers = [RZ_layer, RX_layer,  CRx_layer] * args.nlayers
    params_per_layer = [8, 8, 16] * args.nlayers
    classifier = PQC_cirq(layers=layers, params_per_layer=params_per_layer,
                          n_layers=len(layers), eta=args.eta,
                          sample_grad=args.sample_grad)

    n_epochs = args.niter
    batch_size = args.batch
    if args.loadfile:
        classifier.load_parameters_from_file(args.loadfile)
    test_accuracies = []
    train_accuracies = []

    for epoch in tqdm(range(n_epochs)):
        sel = np.random.choice(len(x_train), batch_size)
        X_batch, Y_batch = x_train[sel], y_train[sel]
        classifier.update(X_batch, Y_batch)
        if (epoch % 20 == 0) and epoch > 0:
            test_accuracies.append(classifier.accuracy(x_test, y_test))
            train_accuracies.append(classifier.accuracy(x_train, y_train))
        if ((epoch+1) % 200 == 0):
            fname = 'weights/layers_{}_eta_{}_epoch_{}_{}'.format(args.nlayers, args.eta, args.epoch_offset + epoch+1, args.suffix)
            classifier.save_parameters(fname)

    print(test_accuracies, train_accuracies)
    classifier.test_accuracies.extend(test_accuracies)
    classifier.train_accuracies.extend(train_accuracies)

    fname = 'classifiers/layers_{}_eta_{}_epoch_{}_sample_grad_{}_{}'.format(args.nlayers,
                                                                         args.eta,
                                                                         args.niter + args.epoch_offset,
                                                                         args.sample_grad,
                                                                         args.suffix)
    classifier.pickle_self(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a PQC on cirq backend.')
    parser.add_argument('--eta', type=float, help='learning rate', default=.01)
    parser.add_argument('--nlayers', type=int, help='number of random layers', default=3)
    parser.add_argument('--niter', type=int, help='number of iterations', default=100)
    parser.add_argument('--sample_grad', type=int, help='batch size', default=-1)
    parser.add_argument('--load', dest='loadfile', help='file with parameter values')
    parser.add_argument('--suffix', dest='suffix', help='add description to filename', default="0")
    parser.add_argument('--offset', dest='epoch_offset', type=int, default=0,
                        help='epoch to continue from (affects saved filename)')

    args = parser.parse_args()
    print(args)
    main(args)
