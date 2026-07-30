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


def do_backprop(target_id, probs, W, x, U, h, H, context_ids):
    # subtract 1 from the entry at the ground-truth index, and leave the rest unchanged.
    dL_dy = probs.copy() # shape: (vocab_size,)
    dL_dy[target_id] -=1 # shape: (vocab_size,)

    # as these are sums, we just distribute back the gradient
    dL_du = dL_dy # shape: (vocab_size,)
    dL_dw = dL_dy # shape: (vocab_size,)
    dL_db = dL_dy   # shape: (vocab_size,)

    dL_dW = [[dL_dw[k] * x[j] for k in range(len(W[0]))] for j in range(len(W))] # shape: (input_dim, vocab_size)
    # partial gradient into x from the residual path only (the hidden path via H is added later)
    dL_dx_residual = [sum(W[j][k] * dL_dw[k] for k in range(len(W[0]))) for j in range(len(W))] # shape: (input_dim,)

    dL_dU = [[dL_du[k] * h[j] for k in range(len(U[0]))] for j in range(len(U))] # shape: (hidden_dim, vocab_size)

    dL_dh = [sum(U[j][k] * dL_du[k] for k in range(len(U[0]))) for j in range(len(U))] # shape: (hidden_dim,)

    dL_dhpre = [g * (1 - hk*hk) for g, hk in zip(dL_dh, h)]  # shape: (hidden_dim,)

    dL_dd = dL_dhpre

    dL_dH = [[dL_dhpre[k] * x[j] for k in range(len(H[0]))] for j in range(len(H))] # shape: (input_dim, hidden_dim)

    dL_dx_hidden = [sum(H[j][k] * dL_dhpre[k] for k in range(len(H[0]))) for j in range(len(H))] # shape: (input_dim,)

    dL_dx = [dL_dx_residual[j] + dL_dx_hidden[j] for j in range(len(dL_dx_residual))] # shape: (input_dim,)

    dL_dC = {}  # {token_id: gradient vector of length embedding_dim}
    for t, token_id in enumerate(context_ids):
        chunk = dL_dx[t * embedding_dim:(t + 1) * embedding_dim]
        if token_id in dL_dC:
            dL_dC[token_id] = [a + b for a, b in zip(dL_dC[token_id], chunk)]
        else:
            dL_dC[token_id] = chunk

    return dL_dC, dL_dH, dL_dd, dL_dW, dL_dU, dL_db

def apply_gradients(C, H, d, W, U, b, grads, lr):
    dL_dC, dL_dH, dL_dd, dL_dW, dL_dU, dL_db = grads

    # update the embedding matrix
    for token_id, grad in dL_dC.items():
        C[token_id] = [c - lr * g for c, g in zip(C[token_id], grad)]

    # update the hidden layer weights
    for j in range(len(H)):
        H[j] = [h - lr * g for h, g in zip(H[j], dL_dH[j])]

    # update the hidden layer bias (in place, so the caller's list is modified)
    d[:] = [d_i - lr * g for d_i, g in zip(d, dL_dd)]

    # update the output layer weights W
    for j in range(len(W)):
        W[j] = [w - lr * g for w, g in zip(W[j], dL_dW[j])]

    # update the output layer weights U
    for j in range(len(U)):
        U[j] = [u - lr * g for u, g in zip(U[j], dL_dU[j])]

    # update the output layer bias (in place, so the caller's list is modified)
    b[:] = [b_i - lr * g for b_i, g in zip(b, dL_db)]


# PARAMS
embedding_dim = 60  # size of the embedding vector for each token
hidden_dim = 50  # size of the hidden layer
context = 5  # number of previous words used to predict the next one
lr = 0.1  # learning rate

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

        grads = do_backprop(target_id, probs, W, x, U, h, H, context_ids)

        apply_gradients(C, H, d, W, U, b, grads, lr)