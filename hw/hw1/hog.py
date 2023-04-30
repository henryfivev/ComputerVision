import cv2
import numpy as np
from skimage.feature import hog
from skimage import data, exposure
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# 加载图像并预处理
img1 = cv2.imread('./images/1/uttower1.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('./images/1/uttower2.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 计算HOG描述子
fd1, hog_image1 = hog(img1, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
fd2, hog_image2 = hog(img2, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

matches = []
for i in range(fd1.shape[0]):
    closest_dist = np.inf
    closest_index = -1
    for j in range(fd2.shape[0]):
        dist = distance.euclidean(fd1[i].reshape(-1), fd2[j].reshape(-1))
        if dist < closest_dist:
            closest_dist = dist
            closest_index = j
    matches.append((i, closest_index))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

print(1)

# 绘制第一个图像和HOG描述子
ax[0].imshow(img1, cmap='gray')
ax[0].axis('off')
ax[0].imshow(hog_image1, alpha=0.5)

# 绘制第二个图像和HOG描述子
ax[1].imshow(img2, cmap='gray')
ax[1].axis('off')
ax[1].imshow(hog_image2, alpha=0.5)

print(2)
print(type(matches))
print(matches[0])

# 绘制匹配线
for i, j in matches:
    point1 = (i % hog_image1.shape[1], i // hog_image1.shape[1])
    point2 = (j % hog_image2.shape[1], j // hog_image2.shape[1] + hog_image1.shape[0])
    line = mlines.Line2D([point1[0], point2[0]], [point1[1], point2[1]], linewidth=1, color='b')
    ax[1].add_line(line)
plt.show()

# 可视化HOG描述子
hog_image_rescaled1 = exposure.rescale_intensity(hog_image1, in_range=(0, 10))
hog_image_rescaled2 = exposure.rescale_intensity(hog_image2, in_range=(0, 10))
cv2.imshow("HOG Image1", hog_image_rescaled1)
cv2.imshow("HOG Image2", hog_image_rescaled2)
cv2.waitKey(0)
cv2.destroyAllWindows()
