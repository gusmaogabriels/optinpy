import jax

# Scientific accuracy tests use x64; explicit float32 tests cover default JAX precision.
jax.config.update('jax_enable_x64', True)
