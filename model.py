import jax
import functools
from jax.scipy.sparse import linalg as sp_linalg
from jax import scipy as sp
from jax import numpy as jnp
from neural_tangents import stax
from tqdm import tqdm

# ==============================================================================
#                 CONFIGURATION FLAG
# ==============================================================================
# Set this to True to use the memory-efficient Conjugate Gradient method.
# Set this to False to use the original direct solve method (will OOM on large datasets).
CG_METHOD = False
# ==============================================================================


def compute_kernel_in_batches(kernel_fn, X1, X2, batch_size=10000):
    """
    Computes the kernel matrix K(X1, X2) in batches to save memory.
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    row_blocks = []
    for i in tqdm(range(0, n1, batch_size), desc="Computing Kernel (Rows)"):
        X1_batch = X1[i : i + batch_size]
        col_blocks = []
        for j in range(0, n2, batch_size):
            X2_batch = X2[j : j + batch_size]
            block = kernel_fn(X1_batch, X2_batch)
            col_blocks.append(block)
        
        row_blocks.append(jnp.hstack(col_blocks))

    return jnp.vstack(row_blocks)



def make_kernelized_rr_forward(hyper_params):
    _, _, base_kernel_fn = FullyConnectedNetwork(
        depth=hyper_params["depth"], num_classes=hyper_params["num_items"]
    )
    batch_size = hyper_params.get('kernel_batch_size', 10000)
    
    # --- OPTION 1: Memory-Efficient Conjugate Gradient (CG) Method ---
    if CG_METHOD:
        print("Using memory-efficient Conjugate Gradient (CG) solver.")
        kernel_fn = functools.partial(base_kernel_fn, get="ntk")

        # ======================================================================
        #                            IMPORTANT NOTE
        # ======================================================================
        # The @jax.jit decorator has been removed from this function.
        #
        # WHY? This function's complexity causes JAX to compile it "eagerly,"
        # which initializes the GPU immediately and conflicts with forking data
        # loaders (like tqdm.pandas), causing a program hang. Removing the
        # decorator makes JAX "lazy," fixing the hang.
        #
        # THE HUGE DOWNSIDE: Without @jit, this function will be EXTREMELY
        # SLOW, as every step will be interpreted by Python instead of being
        # compiled into a single optimized GPU kernel.
        #
        # THE REAL SOLUTION: The best fix is to address the root cause by
        # disabling the forking in your data loader (e.g., by removing the
        # `tqdm.pandas()` line from data.py). This will allow you to safely
        # add `@jax.jit` back to this function for maximum performance.
        # ======================================================================

        # @jax.jit # <--- REMOVED TO PREVENT HANG, BUT SACRIFICES ALL PERFORMANCE
        def kernelized_rr_forward_cg(X_train, X_predict, reg=0.1):
            n_train = X_train.shape[0]
            Y_train = X_train

            # This inner jit is fine and helps a little
            # @jax.jit
            def get_diag(x):
                return kernel_fn(x[None, :], x[None, :])[0, 0]
            trace_K_train = jnp.sum(jax.vmap(get_diag)(X_train))
            lambda_reg = jnp.abs(reg) * trace_K_train / n_train

            def K_reg_matvec(v):
                def body_fun(i, val):
                    start = i * batch_size
                    X_batch = jax.lax.dynamic_slice_in_dim(X_train, start, batch_size, axis=0)
                    update_block = kernel_fn(X_batch, X_train) @ v
                    
                    old_slice = jax.lax.dynamic_slice(val, (start,), update_block.shape)
                    new_slice = old_slice + update_block
                    return jax.lax.dynamic_update_slice(val, new_slice, (start,))
                
                num_batches = (n_train + batch_size - 1) // batch_size
                result_Kv = jax.lax.fori_loop(0, num_batches, body_fun, jnp.zeros(n_train))
                return result_Kv + lambda_reg * v

            solve_one_col = lambda y_col: sp_linalg.cg(K_reg_matvec, y_col)[0]
            alpha = jax.vmap(solve_one_col, in_axes=1, out_axes=1)(Y_train)

            n_predict, n_outputs = X_predict.shape[0], Y_train.shape[1]
            def pred_body_fun(i, y_pred):
                start = i * batch_size
                X_batch = jax.lax.dynamic_slice_in_dim(X_predict, start, batch_size, axis=0)
                update_block = kernel_fn(X_batch, X_train) @ alpha
                return jax.lax.dynamic_update_slice(y_pred, update_block, (start, 0))

            num_pred_batches = (n_predict + batch_size - 1) // batch_size
            Y_predict = jax.lax.fori_loop(0, num_pred_batches, pred_body_fun, jnp.zeros((n_predict, n_outputs)))
            
            return Y_predict
        
        return kernelized_rr_forward_cg, kernel_fn

    # --- OPTION 2: Original Direct Solve Method ---
    else:
        print("Using original direct `linalg.solve` solver (high memory usage).")
        _, _, kernel_fn = FullyConnectedNetwork(
            depth=hyper_params["depth"], num_classes=hyper_params["num_items"]
        )
        # NOTE: Un-comment this if the dataset size is very big (didn't need it for experiments in the paper)
        # kernel_fn = nt.batch(kernel_fn, batch_size=128)
        # kernel_fn = functools.partial(kernel_fn, get="ntk")
        kernel_fn = lambda X1, X2: compute_kernel_in_batches(
            kernel_fn=lambda x1, x2: base_kernel_fn(x1, x2, get='ntk'),
            X1=X1,                                                     
            X2=X2,                                          
            batch_size=batch_size                                    
        )


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
                solution = jnp.linalg.solve(K_reg, X_train, assume_a="pos")
            except:
                # Fallback to a more stable but slower method
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