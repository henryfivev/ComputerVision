import cv2
import numpy as np

# 1. read img as gray scale image
img = cv2.imread("./images/1/sudoku.png")
o = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. get I_x and I_y with sobel
sobelx = np.float32(cv2.Sobel(o, cv2.CV_16S, 1, 0, ksize=3))
# sobelx = cv2.convertScaleAbs(sobelx)
sobely = np.float32(cv2.Sobel(o, cv2.CV_16S, 0, 1, ksize=3))
# sobely = cv2.convertScaleAbs(sobely)

# 3. get I_xI_y
# there are two ways to get I_xI_y
# 1.
# sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# 2.
# sobelxy = cv2.Sobel(o, cv2.CV_64F, 1, 1)
# sobelxy = cv2.convertScaleAbs(sobelxy)
# 3.
# sobelxy = np.reshape(
#     [
#         sobelx[i][j]*sobely[i][j]
#         for i in range(sobelx.shape[0])
#         for j in range(sobelx.shape[1])
#     ],
#     (sobelx.shape[0], sobelx.shape[1]),
# )
# 4.
sobelxy = np.zeros(np.shape(sobelx), dtype=np.float32)
sobelxy[:, :] = sobelx[:, :] * sobely[:, :]

np.savetxt("3sobelx.txt", sobelx)
np.savetxt("3sobely.txt", sobely)
np.savetxt("3sobelxy.txt", sobelxy)

# 4. multiple with w(), to get M matrix
# w() can be all 1 or gaussian filter

# here we set w=1, which means we will do nothing
# but also we set 1 as the size of window, which means window=1x1
# if window size is bigger than 1x1, then we need to sum the Ix, Iy and Ixy up

# 5. calculate the eigenvalue and eigenvector of M
# we use np.linalg.eig(A) to get eigenvalue and eigenvector

windowSize = 3
radius = int((windowSize - 1) / 2)
M = np.zeros([o.shape[0] - windowSize + 1, o.shape[1] - windowSize + 1, 2, 2])
valueM = np.zeros([M.shape[0], M.shape[1], 2])
R = np.zeros([M.shape[0], M.shape[1]])
k = 0.04

M[:, :, 0, 0] = np.reshape(
    [
        (
            (
                np.sum(
                    sobelx[
                        max(0, i - radius) : min(i + radius, sobelx.shape[0]),
                        max(0, j - radius) : min(j + radius, sobelx.shape[1]),
                    ]
                )
            )
            ** 2
        )
        / (windowSize**2)
        for i in range(radius, sobelx.shape[0] - radius)
        for j in range(radius, sobelx.shape[1] - radius)
    ],
    (M.shape[0], M.shape[1]),
)

M[:, :, 1, 1] = np.reshape(
    [
        (
            (
                np.sum(
                    sobely[
                        max(0, i - radius) : min(i + radius, sobely.shape[0]),
                        max(0, j - radius) : min(j + radius, sobely.shape[1]),
                    ]
                )
            )
            ** 2
        )
        / (windowSize**2)
        for i in range(radius, sobelx.shape[0] - radius)
        for j in range(radius, sobelx.shape[1] - radius)
    ],
    (M.shape[0], M.shape[1]),
)
M[:, :, 0, 1] = np.reshape(
    [
        (
            np.sum(
                sobelxy[
                    max(0, i - radius) : min(i + radius, sobelxy.shape[0]),
                    max(0, j - radius) : min(j + radius, sobelxy.shape[1]),
                ]
            )
        )
        / (windowSize**2)
        for i in range(radius, sobelx.shape[0] - radius)
        for j in range(radius, sobelx.shape[1] - radius)
    ],
    (M.shape[0], M.shape[1]),
)
M[:, :, 1, 0] = M[:, :, 0, 1]

valueM, _ = np.linalg.eig(M[:][:])

# 6. calculate R by R = l1*l2 - k(l1+l2)^2
l1 = valueM[:, :, 0]
l2 = valueM[:, :, 1]
R[:, :] = (l1 * l2) - k * ((l1 + l2) ** 2)

# 7. use threshold to judge
R_max = np.max(R)
offset = 1
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i][j] > R_max * 0.01 and R[i][j] > 0:
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
cv2.imwrite("sukodu_out3.png", img)
cv2.waitKey()
cv2.destroyAllWindows()
