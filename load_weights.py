import numpy as np

def load_weight(weights_file):
    with open(weights_file, 'r') as file:
        arch = [int(i) for i in file.readline().split()]

        weights = []
        for ii in range(arch[1]):
            line = file.readline()
            line = [float(i) for i in line.split()]
            weights = weights + [line]
        w1 = np.asarray(weights).T

        weights = []
        for ii in range(arch[2]):
            line = file.readline()
            line = [float(i) for i in line.split()]
            weights = weights + [line]
        w2 = np.asarray(weights).T

        n_inputs = arch[0]
        n_hidden = arch[1]
        n_output = arch[2]
        w1 = w1
        w2 = w2
        return n_inputs, n_hidden, n_output, w1, w2