import random
import math
import logging
import argparse

from helpers import matmul, transpose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# activation_keys = ['context', 'gt', 'x', 'Hx', 'z_hidden', 'a', 'Ua', 'Wx', 'y', 'probs']
# trainable_params = ["C", "H", "d", "U", "W", "b"]

def backward(activations, params):
    # dloss/dy = softmax_i - 1 if i = gt
    # = softmax_i if i != gt

    # i.e. dloss/dy = probs - one_hot(gt) 
    
    one_hot_gt = [1 if i==activations["gt"] else 0 for i in range(len(activations["probs"]))] # shape [V]
    dloss_dy = [p-i for p, i in zip(activations["probs"], one_hot_gt)] # shape [V]

    # dy/db = 1 (since y = b + Wx + Ua)
    # dloss/db = dloss/dy * dy/db = dloss/dy * 1 = dloss/dy

    dloss_db = dloss_dy # shape [V]

    # dy/dW = x (since y = b + Wx + Ua)
    # dloss/dW = dloss/dy * dy/dW = dloss/dy * x

    dloss_dW = [[vi*xi for vi in dloss_dy] for xi in activations["x"]] # shape [n*m, V]

    # dy/dU = a (since y = b + Wx + Ua)
    # dloss/dU = dloss/dy * dy/dU 

    dloss_dU = [[vi*ai for vi in dloss_dy] for ai in activations["a"]] # shape [h, V]

    # dloss/dd = dloss/dy * dy/dUa * dUa/da * da/dz_hidden * dz_hidden/dd

    # dy/dUa = 1 (since y = b + Wx + Ua)
    # dUa/da = U (since Ua = a * U)
    # da/dz_hidden = 1 - a^2 (since a = tanh(z_hidden))
    # dz_hidden/dd = 1 (since z_hidden = d + Hx)

    # dloss/da = dloss/dy * dy/dUa * dUa/da
    dloss_da = matmul(dloss_dy, transpose(params["U"]))[0] # shape [h]

    # a is shape [h], z_hidden is shape [h], so da/dz_hidden is shape [h]
    da_dz_hidden = [1 - ai**2 for ai in activations["a"]]

    # dloss/dd = dloss/da * da/d_zhidden * dz_hidden/dd
    dloss_dz_hidden = [dldai * dadzi for dldai, dadzi in zip(dloss_da, da_dz_hidden)] # shape [h]

    dloss_dd = dloss_dz_hidden # shape [h]

    # dloss/dH = dloss/dy * dy/dUa * dUa/da * da/dz_hidden * dz_hidden/dH
    # = dloss/da * da/d_zhidden * dz_hidden/dH
    # = dloss/dz_hidden * dz_hidden/dH

    # dz_hidden/dH = x

    dloss_dH = [[dldzi * xi for dldzi in dloss_dz_hidden] for xi in activations["x"]] # shape [n*m, h]

    # dloss/dC = (dloss/dy * dy/dUa * dUa/da * da/dz_hidden * dz_hidden/dx * dx/dC) + (dloss/dy * dy/dWx * dWx/dx * dx/dC)
    # = (dloss/dz_hidden * dz_hidden/dx * dx/dC) + (dloss/dy * dy/dWx * dWx/dx * dx/dC)
    # = ((dloss/dz_hidden * dz_hidden/dx) + (dloss/dy * dy/dWx * dWx/dx)) * dx/dC
    # = dloss/dx * dx/dC

    # d_zhidden/dx = H (since z_hidden = d + Hx), shape [n*m, h]
    # dy/dWx = 1 (since y = b + Wx + Ua)
    # dWx/dx = W (since Wx = x * W)

    # term 1, dloss/dz_hidden (shape [h]) * H (shape [n*m, h]) gives shape [n*m]
    term1 = matmul(dloss_dz_hidden, transpose(params["H"]))[0] # shape [n*m]

    # term 2, dloss/dy (shape [V]) * 1 * W (shape [n*m, V]) gives shape [n*m]
    term2 = matmul(dloss_dy, transpose(params["W"]))[0] # shape [n*m]

    dloss_dx = [t1+t2 for t1, t2 in zip(term1, term2)] # shape [n*m]

    # dx/dC is different since x is getting rows from C
    # shape [V, m]

    # create empty matrix, C is shape [V, m]
    dloss_dC = [[0 for _ in range(len(params["C"][0]))] for _ in range(len(params["C"]))]

    # iterate through the context (length n)
    for i, word in enumerate(activations["context"]):
        # iterate through the embedding dim (m)
        for j in range(len(params["C"][0])):
            dloss_dC[word][j] += dloss_dx[i*len(params["C"][0]) + j] # shape [n*m]

    return {
        "b": dloss_db,
        "W": dloss_dW,
        "U": dloss_dU,
        "d": dloss_dd,
        "H": dloss_dH,
        "C": dloss_dC,
    }