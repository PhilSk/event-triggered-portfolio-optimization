import jax.numpy as jnp
import numpy as np
import csv
import os
from jax import grad, jit, vmap
from jax import random


random_key = random.PRNGKey(1)

def f(A, B):
    return jnp.linalg.norm(A - B)

A = jnp.ones((3, 4))
B = random.randint(random_key, (3, 4), 0, 10).astype(float)

f_grad = grad(f)

print(f'{A} \n\n {B} \n\n {f(A, B)} \n\n {f_grad(A, B)}')


# def f(x):
#     return x[0]**2 + x[1]**2

# def P(x):
#     return jnp.clip(x, a_min=np.array((-1.9, -1.1)), a_max=np.array((1.9, 1.1)))

# f_vmap = vmap(f)
# f_grad = vmap(grad(f))

# dim = 2
# N = 10
# k_max = 100000
# eps = 1e-5

# c_0 = 1.
# c_1 = 10.
# c_2 = 1.

# x = 100*random.uniform(random_key, shape=(N, dim))
# print(x)
# x_p = x.copy()
# f_val = f_vmap(x_p)
# x_g = x_p[jnp.argmin(f_val)]

# x_buf = []
# if os.path.exists("data/x.csv"):
#   os.remove("data/x.csv")

# for i in range(k_max):
#    print(f'\n\nStep {i}\nbest f: {x_g}')
#    x_clipped = P(x)
#    dx = -x + x_clipped - f_grad(x_clipped)
#    x_new = np.array(x + eps*dx)

#    f_new = f_vmap(x_new)
#    indices = tuple(f_new < f_val)
#    x_p_new = x_p.copy()
#    x_p_new[indices, :] = x_new[indices, :]
#    x_p = x_p_new
#    func_vals = f_vmap(x_p)
#    g_best_index = jnp.argmin(func_vals)
#    x_g = x_p[g_best_index]

#    print(func_vals[g_best_index])

#    x_buf.append(
#          np.append(x_g, func_vals[g_best_index])
#       )
#    if i % 2 == 0 and i!=0:
#       with open('data/data.csv', 'a') as f: 
#          w = csv.writer(f) 
#          w.writerows(x_buf)
#       x_buf = []

#    (r1, r2) = random.uniform(random_key, shape=(2,))
#    x = x + c_0*(x_new - x) + c_1*r1*(x_p - x) + c_2*r2*(x_g - x)
