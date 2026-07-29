import random
import math
import logging

from helpers import matmul

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

def forward(i:int, train_ids, params):
	activations = {} # dict to store activations for backward pass

	# get input batch
	context, gt = get_training_data(i, params["n"], train_ids)
	activations["context"] = context
	activations["gt"] = gt
	x = get_sentence_embeddings(context, params["C"])
	activations["x"] = x

	Hx = matmul(x, params["H"]) # shape [1, h], we drop the 1 dim
	activations["Hx"] = Hx[0]

	z_hidden = [di+Hxi for di, Hxi in zip(params["d"], Hx[0])]
	activations["z_hidden"] = z_hidden

	a = [math.tanh(x) for x in z_hidden] # shape [h]
	activations["a"] = a

	Ua = matmul(a, params["U"]) # shape [1, V], we drop the 1 dim
	activations["Ua"] = Ua[0]

	# residuals
	Wx = matmul(x, params["W"]) # shape [1, V], we drop the 1 dim
	activations["Wx"] = Wx[0]

	y = [bi+wi+ai for bi, wi, ai in zip(params["b"], Wx[0], Ua[0])]
	activations["y"] = y

	# softmax
	es = [math.exp(i) for i in y]
	sum_es = sum(es)
	probs = [i/sum_es for i in es]
	activations["probs"] = probs

	loss = -math.log(probs[gt])
	logger.info(f"{loss=}")
	return loss, activations