import argparse
import logging

from forward import build_array, forward
from backward import backward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# initial param set up
n=4 # context length
m = 50 # word embedding size
h = 60 # hidden dim
logger.info(f"{n=}, {m=}, {h=}")

# load training data
with open("../data/train.ids", "r") as f:
	train_ids = f.read().split()
train_ids = [int(id) for id in train_ids]

# load vocab
with open("../data/vocab.txt", "r") as f:
	vocab = f.read().split()

V = len(vocab) # vocab size
logger.info(f"{V=}")

# trainable matrixes
C = build_array([V, m]) # embeddings matrix, shape [V, m]
H = build_array([m*n, h]) # shape [n*m, h]
d = build_array([h]) # shape [h]
U = build_array([h, V]) # shape [h, V]
W = build_array([n*m, V]) # shape [n*m, V]
b = build_array(V) # shape [V]

def update_params(params: list, grads: list, lr):
	"""Update params in place using gradient descent

	Parameters
	----------
	param : list
		1 or 2 dim matrix of parameters to update
	grads : list
		1 or 2 dim matrix of gradients (one per each parameter)
	lr : float
		Learning rate
	"""
	assert len(params) == len(grads)

	if isinstance(params[0], list):
		assert len(params[0]) == len(grads[0])

		# iterate over rows
		for param_row, g_row in zip(params, grads):
			# update in place
			for i in range(len(param_row)):
				param_row[i] -= lr * g_row[i]

	else:
		# update in place
		for i in range(len(params)):
			params[i] -= lr * grads[i]


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--n-steps", type=int, default=10, help="number of training steps")
	parser.add_argument("-i", type=int, default=0, help="start index for training data")

	params = {
		"n":n,
		"m":m,
		"h":h,
		"C":C,
		"H":H,
		"d":d,
		"U":U,
		"W":W,
		"b":b,
	}

	args = parser.parse_args()

	for step in range(args.n_steps):
		# forward
		loss, activations = forward(args.i, train_ids, params)

		# backward
		grads = backward(activations, params)

		# update params
		lr = 0.01
		for k, grad in grads.items():
			update_params(
				params=params[k], 
				grads=grad, 
				lr=lr
			)

