import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('./images/1/uttower1.jpg')
img2 = cv2.imread('./images/1/uttower2.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Harris 角点检测
harris1 = cv2.cornerHarris(gray1, 2, 3, 0.04)
harris2 = cv2.cornerHarris(gray2, 2, 3, 0.04)

# 提取角点
corners1 = np.argwhere(harris1 > 0.01 * harris1.max())
corners2 = np.argwhere(harris2 > 0.01 * harris2.max())

kp1 = [
    cv2.KeyPoint(float(x), float(y), 8) for x, y in corners1
]
kp2 = [
    cv2.KeyPoint(float(x), float(y), 8) for x, y in corners2
]
keypoints1 = np.float32([m.pt for m in kp1])
keypoints2 = np.float32([m.pt for m in kp2])


# 提取角点特征
winSize = (2, 2)
blockSize = (2, 2)
blockStride = (1, 1)
cellSize = (1, 1)
nbins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

features1 = []
for i in range(corners1.shape[0]):
    x, y = corners1[i]
    patch = gray1[y-1:y+1, x-1:x+1]
    if patch is None or patch.size == 0:
        continue  # 跳过该 patch
    feature = hog.compute(patch)
    features1.append(feature)
features1 = np.concatenate(features1, axis=0)
features1 = np.float32(features1[:4000])

features2 = []
for i in range(corners2.shape[0]):
    x, y = corners2[i]
    patch = gray2[y-1:y+1, x-1:x+1]
    if patch is None or patch.size == 0:
        continue  # 跳过该 patch
    feature = hog.compute(patch)
    features2.append(feature)
features2 = np.concatenate(features2, axis=0)
features2 = np.float32(features2[:4000])

# 角点匹配
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = matcher.match(features1, features2)
matches = sorted(matches, key=lambda x: x.distance)
img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[: 100], img2, flags=2)

src_pts = np.float32([corners1[m.queryIdx] for m in matches])
dst_pts = np.float32([corners2[m.trainIdx] for m in matches])



H, status = cv2.findHomography(corners2, corners2, cv2.RANSAC, 4)

print(H)

result = cv2.warpPerspective(
            img2, H, (img2.shape[1] + img1.shape[1], img2.shape[0])
        )
result[0 : img1.shape[0], 0 : img1.shape[1]] = img1



# cv2.imshow('image1', img_o1)
# cv2.imshow('image2', img_o2)
cv2.imshow('image_match', img_match)
cv2.imshow('image_stitch', result)
# cv2.imwrite("./output/uttower_match.png", img_o1)
cv2.waitKey(0)
cv2.destroyAllWindows()