import cv2
o = cv2.imread('./images/1/sudoku.png',cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(o,cv2.CV_64F,1,0)
sobelx = cv2.convertScaleAbs(sobelx)   # 转回uint8

sobely = cv2.Sobel(o,cv2.CV_64F,0,1)
sobely = cv2.convertScaleAbs(sobely)   # 转回uint8

# sobelxy =  cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
sobelxy = cv2.Sobel(o,cv2.CV_64F,1,1)
sobelxy = cv2.convertScaleAbs(sobelxy)   # 转回uint8

cv2.imshow("original",o)
cv2.imshow("x",sobelx)
cv2.imshow("y",sobely)

cv2.imshow("xy",sobelxy)

cv2.waitKey()
cv2.destroyAllWindows()
