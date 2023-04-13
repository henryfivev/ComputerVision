import cv2

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


# 5. calculate R
# we use np.linalg.eig(A) to get eigenvalue and eigenvector


cv2.imshow("original", o)
cv2.imshow("x", sobelx)
cv2.imshow("y", sobely)

cv2.imshow("xy", sobelxy)

cv2.waitKey()
cv2.destroyAllWindows()
