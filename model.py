import jax
import functools
from jax import scipy as sp
from jax import numpy as jnp
from neural_tangents import stax
import neural_tangents as nt
from tqdm import tqdm

# def compute_kernel_in_batches(kernel_fn, X1, X2, batch_size=1024):
#     """Computes kernel matrix in memory-safe batches."""
#     n1, n2 = X1.shape[0], X2.shape[0]
#     rows = []

#     for i in tqdm(range(0, n1, batch_size), desc="Batching over X1"):
#         X1_batch = X1[i : i + batch_size]
#         blocks = []
#         for j in range(0, n2, batch_size):
#             X2_batch = X2[j : j + batch_size]
#             block = kernel_fn(X1_batch, X2_batch, get="ntk")
#             blocks.append(block)
#         rows.append(jnp.concatenate(blocks, axis=1))

#     return jnp.concatenate(rows, axis=0)

def make_kernelized_rr_forward(hyper_params):
    _, _, kernel_fn = FullyConnectedNetwork(
        depth=hyper_params["depth"], num_classes=hyper_params["num_items"]
    )
    # NOTE: Un-comment this if the dataset size is very big (didn't need it for experiments in the paper)
    # kernel_fn = nt.batch(kernel_fn, batch_size=138493)
    kernel_fn = functools.partial(kernel_fn, get="ntk")



    @jax.jit
    def kernelized_rr_forward(X_train, X_predict, reg=0.1):
        K_train = kernel_fn(X_train, X_train)
        K_predict = kernel_fn(X_predict, X_train)
        K_reg = (
            K_train
            + jnp.abs(reg)
            * jnp.trace(K_train)
            * jnp.eye(K_train.shape[0])
            / K_train.shape[0]
        )
        # Try using jax.numpy.linalg.solve instead of scipy
        try:
            # solution = jnp.linalg.solve(K_reg, X_train, assume_a="pos")
            solution = jnp.linalg.solve(K_reg, X_train)
        except:
            # Fallback to a more stable but slower method

            print("K_reg shape:", K_reg.shape)
            print("X_train shape:", X_train.shape)

            print("NaNs in K_reg:", jnp.isnan(K_reg).any())
            print("NaNs in X_train:", jnp.isnan(X_train).any())
            print("Infs in K_reg:", jnp.isinf(K_reg).any())
            print("Infs in X_train:", jnp.isinf(X_train).any())
            print("NaNs in K_reg:", jnp.isnan(K_reg).any().item())





            solution = jnp.linalg.lstsq(K_reg, X_train)[0]
        # return jnp.dot(K_predict, sp.linalg.solve(K_reg, X_train, sym_pos=True))
        return jnp.dot(K_predict, solution)
        # return jnp.dot(K_predict, sp.linalg.solve(K_reg, X_train, assume_a='pos'))

    return kernelized_rr_forward, kernel_fn


def FullyConnectedNetwork(
    depth, W_std=2**0.5, b_std=0.1, num_classes=10, parameterization="ntk"
):
    activation_fn = stax.Relu()
    dense = functools.partial(
        stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization
    )

    layers = [stax.Flatten()]
    # NOTE: setting width = 1024 doesn't matter as the NTK parameterization will stretch this till \infty
    for _ in range(depth):
        layers += [dense(1024), activation_fn]
    layers += [
        stax.Dense(
            num_classes, W_std=W_std, b_std=b_std, parameterization=parameterization
        )
    ]

    return stax.serial(*layers)