import numpy as np

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def get_init_weights(layers):
    for i in range(0, len(layers)):
        if i == len(layers) - 1:
            print(layers[i])
        else:
            print(layers[i], end=' ')

    for layer in range(1, len(layers)):
        for node in range(layers[layer]):
            print(' '.join(map(str, np.round(np.random.rand(layers[layer - 1] + 1) / 10, 3))))

get_init_weights([7,20,1])