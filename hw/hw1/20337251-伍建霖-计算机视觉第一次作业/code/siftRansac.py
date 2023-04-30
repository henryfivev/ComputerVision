import cv2
import numpy as np

# 1. read img
img1 = cv2.imread('./yosemite12.png')
img2 = cv2.imread('./yosemite34.png')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# 2. create sift descriptor
sift1 = cv2.SIFT_create()
sift2 = cv2.SIFT_create()
kp1, des1 = sift1.detectAndCompute(gray1, None)
kp2, des2 = sift2.detectAndCompute(gray2, None)
# img_o1 = cv2.drawKeypoints(gray1, kp1, img1)
# img_o2 = cv2.drawKeypoints(gray2, kp2, img2)
# 3. match
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[: 50], img2, flags=2)
# 4. compute ransac
# 获取匹配对的点坐标
kps1 = np.float32([kp.pt for kp in kp1])
kps2 = np.float32([kp.pt for kp in kp2])
pt1 = np.float32([kps1[match.queryIdx] for match in matches])
pt2 = np.float32([kps2[match.trainIdx] for match in matches])

print(np.shape(pt1))
print("1111")
# 计算视角变换矩阵
H, status = cv2.findHomography(pt2, pt1, cv2.RANSAC, 4)

print(H)

result = cv2.warpPerspective(
            img2, H, (img2.shape[1] + img1.shape[1], img2.shape[0])
        )
result[0 : img1.shape[0], 0 : img1.shape[1]] = img1



# cv2.imshow('image1', img_o1)
# cv2.imshow('image2', img_o2)
# cv2.imshow('image_match', img_match)
# cv2.imshow('image_stitch', result)
cv2.imwrite("./yosemite1234.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()