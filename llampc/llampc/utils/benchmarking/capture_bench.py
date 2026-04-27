import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("/home/kathy/jax_cache")

import jax
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
jax.config.update("jax_log_compiles", True)
import jax.numpy as jnp
import time

# Simplified version of your actual computation
bank_params_static = jnp.ones((20000, 10))

# Version 1: Captured
@jax.jit
def integrate_captured(known_params, x0, u):
    def rk4(b_p):
        # b_p is shape (10,), x0 is shape (6,), u is shape (2,)
        # Simplified: just sum them to get a scalar result
        return jnp.sum(b_p) * jnp.sum(x0) * jnp.sum(u)
    return jax.vmap(rk4)(bank_params_static)

# Version 2: Runtime arg
@jax.jit
def integrate_runtime(bank_params, known_params, x0, u):
    def rk4(b_p):
        return jnp.sum(b_p) * jnp.sum(x0) * jnp.sum(u)
    return jax.vmap(rk4)(bank_params)

# Warmup
print("Warming up...")
_ = integrate_captured(jnp.ones(10), jnp.ones(6), jnp.ones(2)).block_until_ready()
_ = integrate_runtime(bank_params_static, jnp.ones(10), jnp.ones(6), jnp.ones(2)).block_until_ready()
print("Warmup complete")

# Benchmark
n_runs = 100

print(f"\nBenchmarking {n_runs} runs...")

start = time.time()
for _ in range(n_runs):
    _ = integrate_captured(jnp.ones(10), jnp.ones(6), jnp.ones(2)).block_until_ready()
captured_time = (time.time() - start) / n_runs * 1000
print(f"Captured: {captured_time:.3f}ms per run")

start = time.time()
for _ in range(n_runs):
    _ = integrate_runtime(bank_params_static, jnp.ones(10), jnp.ones(6), jnp.ones(2)).block_until_ready()
runtime_time = (time.time() - start) / n_runs * 1000
print(f"Runtime arg: {runtime_time:.3f}ms per run")

diff_pct = abs(captured_time - runtime_time) / min(captured_time, runtime_time) * 100
print(f"\nDifference: {diff_pct:.1f}%")

if diff_pct < 10:
    print("✓ Performance is essentially identical (within noise)")
else:
    print(f"⚠ There's a measurable difference")