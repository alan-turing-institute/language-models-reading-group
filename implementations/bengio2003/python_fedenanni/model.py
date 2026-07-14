import math
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def read_vocab(path=DATA_DIR / "vocab.txt"):
    """Read vocab.txt into a list where index == token id."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def read_ids(path=DATA_DIR / "train.ids"):
    """Read a whitespace-separated .ids file into a list of ints."""
    with open(path, encoding="utf-8") as f:
        return [int(tok) for tok in f.read().split()]


def create_embedding_matrix(vocab_size, embedding_dim):
    """Create a random embedding matrix of shape (vocab_size, embedding_dim)."""
    return [[random.uniform(-0.1, 0.1) for _ in range(embedding_dim)] for _ in range(vocab_size)]

def create_layer(input_dim, output_dim):
    """Create a random layer weight matrix of shape (input_dim, output_dim)."""
    return [[random.uniform(-0.1, 0.1) for _ in range(output_dim)] for _ in range(input_dim)]

def create_bias_vector(size):
    """Create a random bias vector of given size."""
    return [random.uniform(-0.1, 0.1) for _ in range(size)]

def do_forward_pass(C, H, d, W, U, b, context_ids):
    # build the context window by concatenating the embeddings of the previous words
    x = []
    for token_id in context_ids:
        x.extend(C[token_id])

    # multiply by the hidden layer weights and add the bias
    h = [sum(x[j] * H[j][k] for j in range(len(x))) + d[k] for k in range(len(H[0]))]

    # tanh activation
    h = [math.tanh(val) for val in h]

    # project to output layer using U  
    u = [sum(h[j] * U[j][k] for j in range(len(h))) for k in range(len(U[0]))]

    # build optional residual connection from x to the output layer using W
    w = [sum(x[j] * W[j][k] for j in range(len(x))) for k in range(len(W[0]))]

    # add the bias
    y = [u[k] + w[k] + b[k] for k in range(len(b))]

    # apply softmax to get probabilities
    m = max(y)
    exps = [math.exp(v - m) for v in y]
    z = sum(exps)
    probs = [e / z for e in exps]

    return probs, x, h 



# PARAMS
embedding_dim = 60  # size of the embedding vector for each token
hidden_dim = 50  # size of the hidden layer
context = 5  # number of previous words used to predict the next one

if __name__ == "__main__":
    vocab = read_vocab()
    ids = read_ids()
    C = create_embedding_matrix(len(vocab), embedding_dim)
    H = create_layer(embedding_dim * context, hidden_dim)
    d = create_bias_vector(hidden_dim)
    W = create_layer(embedding_dim * context, len(vocab))
    U = create_layer(hidden_dim, len(vocab))
    b = create_bias_vector(len(vocab))


    print(f"vocab size: {len(vocab)}")
    print(f"train tokens: {len(ids)}")

    # peek at the first 20 tokens as words
    words = [vocab[i] for i in ids[:20]]
    print("first 20 words:", " ".join(words))

    # train with SGD: one forward pass, one backward pass, one update per position
    lr = 0.1
    for i in range(context, len(ids)):
        context_ids = ids[i - context:i]
        target_id = ids[i]
        probs, x, h = do_forward_pass(C, H, d, W, U, b, context_ids)
        loss = -math.log(probs[target_id])
        print(f"position {i}: loss = {loss}")

        #grads = do_backprop(C, H, d, W, U, b, context_ids, target_id, probs, x, h)
        #apply_gradients(C, H, d, W, U, b, grads, lr)