#Seyun Kim
#ECE 469 Artificial Intelligence
#Project 2

import os
from NeuralNet import NET
from scipy.special import expit as sig

def neural_network_train():
	weights = input('Enter filename of initial neural network: ')
	while not os.path.isfile(weights):
		weights = input('Please enter valid filename: ')

	train_set = input('Enter filename of training set: ')
	while not os.path.isfile(train_set):
		train_set = input('Enter valid filename: ')

	# Request output file location
	output_name = input('Enter filename for results: ')
	while not os.path.isfile(output_name):
		open(output_name, 'wb')

	# Request number of epochs
	n_epochs = int(input('Enter positive integer for number of iterations: '))
	while not n_epochs > 0:
		n_epochs = int(input('Enter positive integer: '))

	# Request learning rate
	lr = float(input('Enter positive number for learning rate: '))
	while not lr > 0.0:
		lr = float(input('Enter positive number: '))

	nn = NET(weights = weights, train_set = train_set, output_name = output_name, n_epochs = n_epochs, l_rate = lr)
	nn.train_network()


def neural_network_test():
	trained_weights = input('Enter filename of trained neural network: ')
	while not os.path.isfile(trained_weights):
		trained_weights = input('Enter valid filename: ')

	# Request text file with test set
	test_set = input('Enter filename of testing set: ')
	while not os.path.isfile(test_set):
		test_set = input('Enter valid filename: ')

	# Output text file name
	output_name = input('Enter filename for results: ')
	while not os.path.isfile(output_name):
		open(output_name, 'wb')

	nn = NET(weights = trained_weights, test_set = test_set, output_name = output_name)
	nn.test_network()


if __name__ == '__main__':
	action = input('Are you training? Type y. If not, type n: ')
	if action == 'y':
		neural_network_train()
	elif action == 'n':
		neural_network_test()
