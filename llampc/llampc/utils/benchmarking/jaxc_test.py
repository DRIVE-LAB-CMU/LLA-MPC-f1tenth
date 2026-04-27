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

# Test with operations that DON'T use cuBLAS
print("Test 1: Element-wise operations (no cuBLAS)")
x = jnp.ones((1000, 1000))
y = x + x  # Simple addition, no GEMM
z = y * 2
result = z.block_until_ready()
print("✓ Element-wise operations work!")

print("\nTest 2: Check cache files")
import subprocess
subprocess.run(['ls', '-lah', '/home/kathy/jax_cache/'])
subprocess.run(['find', '/home/kathy/jax_cache/', '-type', 'f'])