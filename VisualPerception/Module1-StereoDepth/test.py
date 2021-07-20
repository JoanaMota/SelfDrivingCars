import numpy as np

# input two matrices
point = [-0.5, 1.5, 9, 1]
mat1 = [1, 0, 0, 1], [0, -1, 0, 2], [0, 0, -1, 10], [0, 0, 0, 1]
mat2 = ([640, 0, 640, 0], [0, 480, 480, 0], [0, 0, 1, 0], [0, 0, 0, 1])

# This will return matrix product of two array
res = np.dot(mat1, point)
print(res)

mat = np.dot(mat2, mat1)
res2 = np.dot(mat, point)
print(res2)

# print resulted matrix
