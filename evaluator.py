#Seyun Kim ECE469 Project 2

import numpy as np

def calculate_metrics(results):
	accuracy = np.asarray((results[:, 0] + results[:, 3]) / np.sum(results, 1), dtype=np.float64)
	precision = np.asarray((results[:, 0]) / np.sum(results[:, 0:2], 1), dtype=np.float64)
	recall = np.asarray((results[:, 0]) / (results[:, 0] + results[:, 2]), dtype=np.float64)
	f1 = np.asarray(2 * precision * recall / (precision + recall), dtype=np.float64)
	return accuracy, precision, recall, f1