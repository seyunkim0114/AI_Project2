import numpy as np
from scipy.special import expit as sig
from sklearn.metrics import confusion_matrix
from evaluator import calculate_metrics
from load_weights import load_weight


class NET:
	def __init__(self, weights=None, train_set=None, test_set=None, output_name=None, n_epochs=None, l_rate=None):
		self.train_set = train_set
		self.test_set = test_set
		self.weights = weights
		self.output_name = output_name
		self.n_epochs = n_epochs
		self.l_rate = l_rate
		self.n_inputs = None
		self.n_hidden = None
		self.n_output = None
		self.w1 = self.w2 = None

		self.a1 = self.a2 = self.a3 = None
		self.ins2 = self.ins3 = None

		self.load_initial_weights()

		if self.train_set is not None:
			self.X_train, self.y_train = self.get_data(self.train_set)
		else:
			self.X_train, self.y_train = None, None

		if self.test_set is not None:
			self.X_test, self.y_test = self.get_data(self.test_set)
		else:
			self.X_test, self.y_test = None, None

	def get_data(self, file_name):
		data = []
		with open(file_name, 'r') as file:
			temp = file.readline().split()
			n_inputs = int(temp[1])
			n_outputs = int(temp[2])

			if n_inputs != self.n_inputs:
				raise ValueError('Incorrect input dimensions.')
			elif n_outputs != self.n_output:
				raise ValueError('Incorrect output dimensions.')

			for line in file:
				line = [float(i) for i in line.split()]
				data = data + [line]

			x = np.asarray(data).T[:n_inputs].T
			y = np.asarray(data).T[n_inputs:].T

			return x, y
		return None, None

	def forward_prop(self, x):
		if len(x.shape) != 2 or x.shape[1] != self.n_inputs:
			raise ValueError('Incorrect input shape ' + str(x.shape) + ' given!')
		else:
			x = np.append(-np.ones((len(x), 1)), x, axis=1)
			self.a1 = x
			self.ins2 = np.matmul(self.a1, self.w1)
			self.a2 = sig(self.ins2)
			self.a2 = np.append(-np.ones((len(self.a2), 1)), self.a2, axis=1)
			self.ins3 = np.matmul(self.a2, self.w2)
			self.a3 = sig(self.ins3)
		return self.a3

	def load_initial_weights(self):
		self.n_inputs, self.n_hidden, self.n_output, self.w1, self.w2 = load_weight(self.weights)

	def test_network(self):
		original_results = np.zeros((self.n_output, 4))
		y_hat = np.round(self.forward_prop(self.X_test), 0)
		for ii in range(self.n_output):
			original_results[ii, :] = np.reshape(confusion_matrix(self.y_test[:, ii], y_hat[:, ii]), 4)
			original_results[ii, 0], original_results[ii, 3] = original_results[ii, 3], original_results[ii, 0]

		original_results = np.asarray(original_results, dtype=np.int32)
		acc, precision, recall, f1 = calculate_metrics(original_results)

		acc = np.expand_dims(acc, 0).T
		precision = np.expand_dims(precision, 0).T
		recall = np.expand_dims(recall, 0).T
		f1 = np.expand_dims(f1, 0).T

		results = np.concatenate((original_results, acc, precision, recall, f1), 1)

		temp = np.average(results[:, 4:], axis=0)
		temp[3] = 2 * temp[1] * temp[2] / (temp[1] + temp[2])

		with open(self.output_name, 'wb') as f:
			for ii in range(results.shape[0]):
				temp_str = '%d %d %d %d %0.3f %0.3f %0.3f %0.3f\n' % tuple(results[ii, :])
				f.write(temp_str.encode('utf-8'))
			temp_str = '%0.3f %0.3f %0.3f %0.3f\n' % calculate_metrics(np.sum(original_results, axis=0, keepdims=True))
			f.write(temp_str.encode('utf-8'))
			temp_str = '%0.3f %0.3f %0.3f %0.3f\n' % tuple(temp)
			f.write(temp_str.encode('utf-8'))

		# Return value not used in most cases
		return np.average(acc)

	def train_network(self):

		for _ in range(self.n_epochs):
			for ii in range(len(self.X_train)):
				temp_x = self.X_train[ii:ii + 1, :]
				temp_y = self.y_train[ii:ii + 1, :]
				self.forward_prop(temp_x)
				delta3 = dsig(self.ins3) * (temp_y - self.a3)
				delta2 = dsig(self.ins2) * np.matmul(delta3, self.w2[1:, ].T)
				self.w2 += self.l_rate * np.matmul(self.a2.T, delta3)
				self.w1 += self.l_rate * np.matmul(self.a1.T, delta2)

		with open(self.output_name, 'wb') as f:
			temp_str = '%d %d %d\n' % (self.n_inputs, self.n_hidden, self.n_output)
			f.write(temp_str.encode('utf-8'))

			np.savetxt(f, self.w1.T, '%0.3f', delimiter=' ')

			np.savetxt(f, self.w2.T, '%0.3f', delimiter = ' ')

def dsig(x):
	return sig(x) * (1 - sig(x))
