import random
import math
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_array(
	shape: list[int] | int, 
	fill_value: callable = lambda: random.uniform(-0.1, 0.1)
):
	if isinstance(shape, int):
		shape = [shape]

	# e.g. shape = 10  should be [x for i in range(len(10))]
	# e.g. shape = [m, n] should be m rows and n cols i.e. [[x for j in range(n)] for i in range(m)]

	def build(dims):
		if len(dims) == 1:
			return [fill_value() for _ in range(dims[0])]
		return [build(dims[1:]) for _ in range(dims[0])]

	return build(shape)

def get_training_data(i: int, n: int, token_ids: list):
	"""Returns training data and ground truth data from a list of tokens.

	Parameters
	----------
	i : int
		start index
	n : int
		context length
	token_ids : list
		list of token ids (e.g. train_ids)
	"""
	return token_ids[i:i+n], token_ids[i+n]

def get_sentence_embeddings(context: list, C: list):
	"""Get sentence embeddings for a list of n token ids using C embeddings matrix.

	Parameters
	----------
	context : list
		list of tokens ids
	C : list
		embeddings matrix
	"""
	x = []

	for word in context:
		x.extend(C[word])

	return x

def matmul(m1: list, m2: list):
	"""
	Parameters
	----------
	m1 : list
		m1 matrix, must be a row vector (1 dim list) or 2 dim matrix (list of lists)
	m2 : list
		m2 matrix, must be a column vector (1 dim list) or 2 dim matrix (list of lists)
	"""
	if not isinstance(m1[0], list): # assume row vector
		m1 = [m1]
	if not isinstance(m2[0], list): # assume column vector
		m2 = [[x] for x in m2]

	o = [] # m1.shape[1], m2.shape[0]

	for i in range(len(m1)):
		o.append([])

		for j in range(len(m2[0])):
			# extract cols from m2
			col = []
			for row in m2:
				col.append(row[j])

			assert len(col) == len(m1[i])

			# do the matrix multiplication
			sum = 0
			for xi, xj in zip(col, m1[i]):
				sum += xi * xj

			# store output
			o[i].append(sum)
		assert len(o[i]) == len(m2[0])
	assert len(o) == len(m1)
	return o

def forward(i:int):
	activations = {} # dict to store activations for backward pass

	# get input batch
	context, gt = get_training_data(i, n, train_ids)
	activations["context"] = context
	activations["gt"] = gt
	x = get_sentence_embeddings(context, C)
	activations["x"] = x

	Hx = matmul(x, H) # shape [1, h], we drop the 1 dim
	activations["Hx"] = Hx[0]

	z_hidden = [di+Hxi for di, Hxi in zip(d, Hx[0])]
	activations["z_hidden"] = z_hidden

	a = [math.tanh(x) for x in z_hidden] # shape [h]
	activations["a"] = a

	Ua = matmul(a, U) # shape [1, V], we drop the 1 dim
	activations["Ua"] = Ua[0]

	Wx = matmul(x, W) # shape [1, V], we drop the 1 dim
	activations["Wx"] = Wx[0]

	y = [bi+wi+ai for bi, wi, ai in zip(b, Wx[0], Ua[0])]
	activations["y"] = y

	# softmax
	es = [math.exp(i) for i in y]
	sum_es = sum(es)
	probs = [i/sum_es for i in es]
	activations["probs"] = probs

	loss = -math.log(probs[gt])
	logger.info(f"{loss=}")
	return loss, activations

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
H = build_array([m*n, h]) # shape [m*n, h]
d = build_array([h]) # shape [h]
U = build_array([h, V]) # shape [h, V]
W = build_array([n*m, V]) # shape [n*m, V]
b = build_array(V) # shape [V]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", type=int, default=0)

	args = parser.parse_args()
	forward(args.i)