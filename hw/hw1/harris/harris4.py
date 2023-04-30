import cv2
import numpy as np

img = cv2.imread("./images/1/sudoku.png")
o = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobelx = np.float32(cv2.Sobel(o, cv2.CV_16S, 1, 0, ksize=3))
sobely = np.float32(cv2.Sobel(o, cv2.CV_16S, 0, 1, ksize=3))
sobelxy = np.zeros(np.shape(sobelx), dtype=np.float32)
sobelxy = sobelx * sobely

k = 0.04
windowSize = 3

m = np.zeros((o.shape[0], o.shape[1], 3), dtype=np.float32)

m[:, :, 0] = cv2.GaussianBlur(
    sobelx[:, :] ** 2, ksize=(windowSize, windowSize), sigmaX=2
)
m[:, :, 1] = cv2.GaussianBlur(
    sobely[:, :] ** 2, ksize=(windowSize, windowSize), sigmaX=2
)
m[:, :, 2] = cv2.GaussianBlur(sobelxy, ksize=(windowSize, windowSize), sigmaX=2)

m = [
    np.array([[m[i, j, 0], m[i, j, 2]], [m[i, j, 2], m[i, j, 1]]])
    for i in range(o.shape[0])
    for j in range(o.shape[1])
]

D, T = list(map(np.linalg.det, m)), list(map(np.trace, m))
R = np.array([d - k * t**2 for d, t in zip(D, T)])

R_max = np.max(R)
R = R.reshape(o.shape[0], o.shape[1])

radius = 1
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i][j] > R_max * 0.01 and R[i][j] > 0:
            img[i][j] = (0, 0, 255)
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
        #     img[i, j] = (0, 0, 255)


# 8. show img
# cv2.imshow("result", img)
# cv2.imshow("x", sobelx)
# cv2.imshow("y", sobely)
# cv2.imshow("xy", sobelxy)
cv2.imwrite("sukodu_out4.png", img)
cv2.waitKey()
cv2.destroyAllWindows()
