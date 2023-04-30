import cv2
import numpy as np

# 1. read img as gray scale image
img = cv2.imread("../images/1/sudoku.png")
o = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. get I_x and I_y with sobel
sobelx = np.float32(cv2.Sobel(o, cv2.CV_16S, 1, 0, ksize=3))
sobely = np.float32(cv2.Sobel(o, cv2.CV_16S, 0, 1, ksize=3))
sobelxy = np.zeros(np.shape(sobelx), dtype=np.float32)
sobelxy = sobelx * sobely

# 4. multiple with w(), to get M matrix


# 5. calculate the eigenvalue and eigenvector of M
# we use np.linalg.eig(A) to get eigenvalue and eigenvector

M = np.zeros([o.shape[0], o.shape[1], 3])
valueM = np.zeros([M.shape[0], 2])
R = np.zeros([o.shape[0], o.shape[1]])
k = 0.04

M[:, :, 0] = np.reshape(
    [x**2 for row in sobelx for x in row], (M.shape[0], M.shape[1])
)
M[:, :, 1] = np.reshape(
    [x**2 for row in sobely for x in row], (M.shape[0], M.shape[1])
)
M[:, :, 2] = np.reshape([x for row in sobelxy for x in row], (M.shape[0], M.shape[1]))

M[:, :, 0] = cv2.GaussianBlur(M[:, :, 0], ksize=(3, 3), sigmaX=2)
M[:, :, 1] = cv2.GaussianBlur(M[:, :, 1], ksize=(3, 3), sigmaX=2)
M[:, :, 2] = cv2.GaussianBlur(M[:, :, 2], ksize=(3, 3), sigmaX=2)
M = [
    np.array([[M[i, j, 0], M[i, j, 2]], [M[i, j, 2], M[i, j, 1]]])
    for i in range(o.shape[0])
    for j in range(o.shape[1])
]
M = np.reshape(M, (o.shape[0], o.shape[1], 2, 2))

valueM, _ = np.linalg.eig(M[:, :])

# 6. calculate R by R = l1*l2 - k(l1+l2)^2
l1 = valueM[:, :, 0]
l2 = valueM[:, :, 1]
R[:, :] = (l1 * l2) - k * ((l1 + l2) ** 2)

# 7. use threshold to judge
R_max = np.max(R)
offset = 0
radius = 1
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i][j] > R_max * 0.1 and R[i][j] > 0:
            img[i + offset][j + offset] = (0, 0, 255)
        # 极大化抑制
        # if (
        #     R[i][j] > 0
        #     and R[i][j] > R.max() * 0.1
        #     and R[i, j]
        #     == np.max(
        #         R[
        #             max(0, i - radius) : min(i + radius, R.shape[0]),
        #             max(0, j - radius) : min(j + radius, R.shape[1]),
        #         ]
        #     )
        # ):
        #     img[i + offset, j + offset] = (0, 0, 255)


img[237][278] = (0, 255, 0)
img[240][280] = (0, 255, 0)
img[250][290] = (0, 255, 0)

print(R[237][278])
print(valueM[237][278])
print("-----")
print(R[240][279])
print(valueM[240][279])
print("-----")
print(R[250][289])
print(valueM[250][289])
print("-----")
print(M[240][279])

# 8. show img
# cv2.imshow("result", img)
# cv2.imshow("x", sobelx)
# cv2.imshow("y", sobely)
# cv2.imshow("xy", sobelxy)
cv2.imwrite("./output/sukodu_keypoint_final.png", img)
cv2.waitKey()
cv2.destroyAllWindows()
