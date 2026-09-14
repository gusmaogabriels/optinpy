"""Run a small JAX optimization example: python main.py."""
import jax
import jax.numpy as jnp

import optinpy


def main():
    jax.config.update('jax_enable_x64', True)
    rosenbrock = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    for method in ('bfgs', 'lbfgs', 'newton'):
        result = optinpy.minimize(rosenbrock, jnp.array([-1.2, 1.]), method=method)
        if not result['success']:
            raise RuntimeError(f'{method} failed with status {result["status"]}')
        print(f'{method}: x={result["x"]}, f={float(result["f"]):.3e}, iterations={int(result["iterations"])}')


if __name__ == '__main__':
    main()
