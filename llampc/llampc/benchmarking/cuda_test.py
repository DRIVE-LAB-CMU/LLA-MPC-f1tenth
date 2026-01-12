def main():
    import jax
    import jax.numpy as jnp
    import time
    
    print("="*60)
    print("JAX GPU TEST")
    print("="*60)
    
    # List all available devices
    devices = jax.devices()
    print("\nAvailable devices:", devices)
    
    # Check the default device
    default_backend = jax.default_backend()
    print("Default backend:", default_backend)
    
    # Example: Verify if GPU (CUDA) is available
    if any(device.platform == "gpu" for device in devices):
        print("✅ JAX detected CUDA-enabled GPU.")
    else:
        print("❌ No GPU detected, using CPU only")
        return
    
    print("\n" + "="*60)
    print("GPU ALLOCATION TESTS")
    print("="*60)
    
    # Test 1: Simple array allocation
    print("\n[Test 1] Simple GPU allocation...")
    try:
        x = jnp.array([1.0, 2.0, 3.0])
        print(f"✓ Created array on device: {x.device()}")
        print(f"  Array: {x}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return
    
    # Test 2: Larger allocation
    print("\n[Test 2] Large array allocation (100 MB)...")
    try:
        size = (5000, 5000)  # ~100 MB for float32
        start = time.time()
        y = jnp.ones(size)
        y.block_until_ready()  # Wait for GPU operation to complete
        elapsed = time.time() - start
        print(f"✓ Created {size} array in {elapsed:.3f}s")
        print(f"  Device: {y.device()}")
        print(f"  Shape: {y.shape}, Size: {y.nbytes / 1e6:.1f} MB")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return
    
    # Test 3: Stack operation (like your code)
    print("\n[Test 3] Stack operation (similar to param_bank)...")
    try:
        arrays = [jnp.ones(10000) for _ in range(10)]
        start = time.time()
        stacked = jnp.stack(arrays, axis=0)
        stacked.block_until_ready()
        elapsed = time.time() - start
        print(f"✓ Stacked 10 arrays in {elapsed:.3f}s")
        print(f"  Result shape: {stacked.shape}")
        print(f"  Device: {stacked.device()}")
    except Exception as e:
        print(f"✗ Stack failed: {e}")
        return
    
    # Test 4: JIT compilation
    print("\n[Test 4] JIT compilation test...")
    try:
        @jax.jit
        def simple_op(x):
            return x * 2 + 1
        
        test_input = jnp.array([1.0, 2.0, 3.0])
        
        # First call - triggers compilation
        start = time.time()
        result = simple_op(test_input)
        result.block_until_ready()
        compile_time = time.time() - start
        
        # Second call - uses cached version
        start = time.time()
        result = simple_op(test_input)
        result.block_until_ready()
        cached_time = time.time() - start
        
        print(f"✓ JIT compilation successful")
        print(f"  First call (compile): {compile_time:.4f}s")
        print(f"  Second call (cached): {cached_time:.4f}s")
        print(f"  Speedup: {compile_time/cached_time:.1f}x")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ JIT failed: {e}")
        return
    
    # Test 5: Matrix multiplication (compute test)
    print("\n[Test 5] Matrix multiplication (compute test)...")
    try:
        A = jnp.ones((1000, 1000))
        B = jnp.ones((1000, 1000))
        
        start = time.time()
        C = jnp.dot(A, B)
        C.block_until_ready()
        elapsed = time.time() - start
        
        print(f"✓ Matrix multiply (1000x1000) in {elapsed:.4f}s")
        print(f"  Result sum: {jnp.sum(C):.0f} (expected: 1000000)")
    except Exception as e:
        print(f"✗ Matrix multiply failed: {e}")
        return
    
    # Test 6: Large stack (stress test - similar to your failing case)
    print("\n[Test 6] Large stack operation (stress test)...")
    try:
        num_arrays = 10
        array_size = 100000
        
        print(f"  Creating {num_arrays} arrays of size {array_size}...")
        arrays = [jnp.ones(array_size) for _ in range(num_arrays)]
        
        print(f"  Stacking...")
        start = time.time()
        large_stack = jnp.stack(arrays, axis=0)
        large_stack.block_until_ready()
        elapsed = time.time() - start
        
        print(f"✓ Large stack completed in {elapsed:.3f}s")
        print(f"  Result shape: {large_stack.shape}")
        print(f"  Size: {large_stack.nbytes / 1e6:.1f} MB")
        
        # Try axis=1 (like your original code)
        print(f"  Testing stack with axis=1...")
        start = time.time()
        large_stack_axis1 = jnp.stack(arrays, axis=1)
        large_stack_axis1.block_until_ready()
        elapsed = time.time() - start
        print(f"✓ Stack axis=1 completed in {elapsed:.3f}s")
        print(f"  Result shape: {large_stack_axis1.shape}")
        
    except Exception as e:
        print(f"✗ Large stack failed: {e}")
        return
    
    # Test 7: Architecture-specific check
    print("\n[Test 7] Architecture compatibility check...")
    try:
        # This will hang if there's an architecture mismatch
        @jax.jit
        def complex_kernel(x):
            # Operations that trigger architecture-specific code
            y = jnp.sin(x) * jnp.cos(x)
            z = jnp.sum(y ** 2)
            return z
        
        test_data = jnp.linspace(0, 10, 10000)
        
        print("  Running complex kernel (may hang with arch mismatch)...")
        start = time.time()
        result = complex_kernel(test_data)
        result.block_until_ready()
        elapsed = time.time() - start
        
        print(f"✓ Complex kernel completed in {elapsed:.4f}s")
        print(f"  Result: {result:.6f}")
        
    except Exception as e:
        print(f"✗ Complex kernel failed: {e}")
        return
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - GPU is working correctly!")
    print("="*60)

if __name__ == '__main__':
    main()
