import cv2
import numpy as np
from skimage.feature import hog

# from sklearn.metrics.pairwise import cosine_similarity
from skimage import transform as tf

# 读取输入图像并转换为灰度图像
img1 = cv2.imread("./images/1/uttower1.jpg")
img2 = cv2.imread("./images/1/uttower2.jpg")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 使用 HOG 描述子提取关键点和其特征
hog_descriptor1 = hog(
    gray1,
    orientations=9,
    pixels_per_cell=(32, 32),
    cells_per_block=(3, 3),
    visualize=False,
)
kp1 = [
    cv2.KeyPoint(x, y, 8) for y in range(gray1.shape[0]) for x in range(gray1.shape[1])
]
keypoints1 = np.float32([m.pt for m in kp1])
features1 = np.float32(hog_descriptor1.reshape(-1, 1))

hog_descriptor2 = hog(
    gray2,
    orientations=9,
    pixels_per_cell=(32, 32),
    cells_per_block=(3, 3),
    visualize=False,
)
kp2 = [
    cv2.KeyPoint(x, y, 8) for y in range(gray2.shape[0]) for x in range(gray2.shape[1])
]
keypoints2 = np.float32([m.pt for m in kp2])
features2 = np.float32(hog_descriptor2.reshape(-1, 1))

# 对提取的关键点进行匹配
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = matcher.match(features1, features2)
matches = sorted(matches, key=lambda x: x.distance, reverse=False)
img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[1900:1950], img2, flags=2)
# 使用 RANSAC 算法找到关键点对应的匹配点对之间的映射关系
src_pts = np.float32([keypoints1[m.queryIdx] for m in matches])
dst_pts = np.float32([keypoints2[m.trainIdx] for m in matches])
M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4)

print(matches[0])
print(np.shape(matches))
print(M)

# 计算映射矩阵，将图像转换为全景图
result = cv2.warpPerspective(img2, M, (img2.shape[1] + img1.shape[1], img2.shape[0]))
result[0 : img1.shape[0], 0 : img1.shape[1]] = img1

# 保存输出图像
cv2.imshow('image_match', img_match)
cv2.imshow('image_stitch', result)
# cv2.imwrite("./output/uttower_match.png", img_o1)
cv2.waitKey(0)
cv2.destroyAllWindows()