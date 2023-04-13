import numpy as np

mat = np.array([[-1, 1, 0],
              [-4, 3, 0],
              [1, 0, 2]])

a, b = np.linalg.eig(mat)

print("a = {}".format(a))
print("b = {}".format(b))