import random
import math
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# activation_keys = ['context', 'gt', 'x', 'Hx', 'z_hidden', 'a', 'Ua', 'Wx', 'y', 'probs']
# trainable_params = ["C", "H", "d", "U", "W", "b"]

def backward(loss, activations, params):
    # dloss/dy = probs - one_hot(gt) 
    
    one_hot_gt = [1 if i==activations["gt"] else 0 for i in range(params["V"])] # shape [V]

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
    dloss_da = [
        sum(dldyi * ui for dldyi, ui in zip(dloss_dy, u_row))
        for u_row in params["U"]
    ] # shape [h]

    # a is shape [h], z_hidden is shape [h], so da/dz_hidden is shape [h]
    da_dz_hidden = [1 - ai**2 for ai in activations["a"]]

    # dloss/dd = dloss/da * da/d_zhidden * dz_hidden/dd
    dloss_dz_hidden = [dldai * dadzi for dldai, dadzi in zip(dloss_da, da_dz_hidden)] # shape [h]

    dloss_dd = dloss_dz_hidden # shape [h]

    # dloss/dH = dloss/dy * dy/dUa * dUa/da * da/dz_hidden * dz_hidden/dH
    # = dloss/da * da/d_zhidden * dz_hidden/dH
    # = dloss/dz_hidden * dz_hidden/dH

    # dz_hidden/dH = x

    dloss_dH = [[dldzi * xi for dldzi in dloss_dz_hidden] for xi in activations["x"]] # shape [n*m, V]

    # dloss/dC = dloss/dx * dx/dC
    