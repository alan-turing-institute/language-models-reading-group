import argparse
import logging

from .forward import build_array, forward
from .backward import backward

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", type=int, default=0)

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
	loss, activations = forward(args.i, train_ids, params)

	backward(loss, activations, train_ids, params)