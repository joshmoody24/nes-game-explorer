"""Is JAX on the GPU, and how fast is a big matrix multiply? Usage: python -m checks.gpu_check"""
import time

import jax
import jax.numpy as jnp

print("jax", jax.__version__, "| devices:", jax.devices())
x = jnp.ones((4096, 4096))
y = (x @ x).block_until_ready()
started = time.perf_counter()
for _ in range(20):
    y = (x @ x).block_until_ready()
seconds = (time.perf_counter() - started) / 20
print(f"4096x4096 matmul: {seconds * 1e3:.1f} ms = {2 * 4096**3 / seconds / 1e12:.1f} TFLOP/s; result ok: {float(y[0, 0]) == 4096.0}")
