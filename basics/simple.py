import jax
import jax.numpy as jnp
from timeit import timeit
print("Using jax", jax.__version__)


if __name__ == "__main__":
    # creating a zeros array
    a = jnp.zeros((2, 5), dtype=jnp.float32)
    print (a)

    # checking the device
    print (f'Device of a: {a.devices()}')

    # jax is immutable
    try:
        a[0][0] = 1
    except Exception as _:
        print ('This should not work due to immutability!')

    a = a.at[0, 0].set(1) # how to change values of a jax array
    print (a)

    # Generating random numbers
    rng = jax.random.PRNGKey(42)
    a = jax.random.normal(rng)
    b = jax.random.normal(rng)
    assert a == b
    print (a, b)

    # SAMPLING: 
    # sampling
    rng = jax.random.PRNGKey(42)
    a = jax.random.normal(rng)

    # if we want to sample twice, split the rng key n + 1 times
    rng, k1, k2 = jax.random.split(rng, num=3)

    a = jax.random.normal(k1)
    b = jax.random.normal(k2)
    assert a != b
    print (a, b)

    # now to sample twice again
    rng, k1, k2 = jax.random.split(rng, num=3)
    c = jax.random.normal(k1) # this will not be equal to a
    d = jax.random.normal(k2) # this will not be equal to b
    print (a, b, c, d)

    def f(x):
        return 3 * x **2 + 2 *x + 1
    
    # computing the gradient
    f_prime = jax.grad(f) # 6x + 2
    print (f_prime(1.0))

    v_and_grad = jax.value_and_grad(f)
    print (v_and_grad(1.0))

    # jaxspr
    from jax import jit
    f_jit = jit(v_and_grad) # jitted function
    _ = f_jit(1.0) # this will compile the function

    # timing the function
    t1, t2 = (
        timeit(lambda: v_and_grad(1.0), number=1000),
        timeit(lambda: f_jit(1.0), number=1000)
    )
    print (f'Without jit: {t1}, with jit: {t2}')



