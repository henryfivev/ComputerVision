import cv2
import numpy as np

# 1. read img as gray scale image
o = cv2.imread("./images/1/sudoku.png", cv2.IMREAD_GRAYSCALE)

# 2. get I_x and I_y with sobel
sobelx = cv2.Sobel(o, cv2.CV_64F, 1, 0)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(o, cv2.CV_64F, 0, 1)
sobely = cv2.convertScaleAbs(sobely)

# 3. get I_xI_y
# there are two ways to get I_xI_y
# sobelxy =  cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
sobelxy = cv2.Sobel(o, cv2.CV_64F, 1, 1)
sobelxy = cv2.convertScaleAbs(sobelxy)

# 4. multiple with w(), to get M matrix
# w() can be all 1 or gaussian filter

# here we set w=1, which means we will do nothing
# but also we set 1 as the size of window, which means window=1x1

# 5. calculate the eigenvalue and eigenvector of M
# we use np.linalg.eig(A) to get eigenvalue and eigenvector

M = np.zeros([o.shape[0], o.shape[1], 2, 2])
valueM = np.zeros([o.shape[0], o.shape[1], 2, 2])
R = np.zeros(o.shape)
k = 0.04
for i in range(o.shape[0]):
    for j in range(o.shape[1]):
        xx = sobelx[i][j]**2
        yy = sobely[i][j]**2
        xy = sobelxy[i][j]

        M[i][j] = np.array([[xx, xy], [xy, yy]])
        valueM[i][j], _ = np.linalg.eig(np.array([[xx, xy], [xy, yy]]))

# 6. calculate R by R = l1*l2 - k(l1+l2)^2
        l1 = valueM[i][j][0][0]
        l2 = valueM[i][j][0][1]
        R[i][j] = l1*l2 - k*(l1+l2)**2

b = input()

# 7. use threshold to judge

# img[dst>0.01*dst.max()]=[0,0,255]
for i in range(o.size):
    if(R[i] > R.max*0.01):
        o[i] = [0, 0, 255]

cv2.imshow("result", o)
cv2.imshow("i", sobelx)
cv2.imshow("y", sobely)
cv2.imshow("xy", sobelxy)

cv2.waitKey()
cv2.destroyAllWindows()
