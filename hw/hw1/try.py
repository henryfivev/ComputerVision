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

R = np.array([[1, 1, 2, 3], [1, 2, 3, 4], [1, 3, 9, 1], [1, 3, 2, 1]])

i = 0
j = 0
p = np.max(R[max(0,i-3):min(i+3,562),max(0,j-3):min(j+3,557)])

print(p.index(9))
print(p)