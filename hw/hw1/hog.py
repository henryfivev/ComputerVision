# 导入cv2模块
import cv2

# 创建HOG对象
hog = cv2.HOGDescriptor()

# 读取图像
im = cv2.imread('sample.jpg')

# 计算HOG特征
h = hog.compute(im)

# 打印HOG特征的维度
print(h.shape)