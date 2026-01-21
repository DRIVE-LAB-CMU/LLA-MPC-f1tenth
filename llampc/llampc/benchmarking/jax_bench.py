import jax
print("JAX version:", jax.__version__)
print("JAX location:", jax.__file__)

# Check if compilation cache module exists
try:
    from jax._src import compilation_cache
    print("✓ compilation_cache module found")
except ImportError as e:
    print(f"✗ compilation_cache module missing: {e}")

# Check all available config flags
print("\nAll JAX config flags:")
import jax.config
print([x for x in dir(jax.config) if 'cache' in x.lower()])