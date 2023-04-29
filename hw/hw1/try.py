import numpy as np

mat = np.array([[-1, 1, 0],
              [-4, 3, 0],
              [1, 0, 2]])

a, b = np.linalg.eig(mat)

print("a = {}".format(a))
print("b = {}".format(b))

list_2d = list([[1, 2], [3, 4]])

squared_numbers = [x**2 for row in list_2d for x in row]
M = np.zeros([2, 2, 2])
valueM = np.zeros(np.shape(M))
print(np.shape(valueM))
print(squared_numbers)