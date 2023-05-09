import numpy as np
import cv2
import matplotlib.pyplot as plt

def pca_image(image, n_components):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 将图像矩阵展平
    flattened_img = gray_image.flatten()
    
    # 数据归一化
    normalized_img = (flattened_img - np.mean(flattened_img)) / np.std(flattened_img)
    
    # 计算协方差矩阵
    covariance_matrix = np.cov(normalized_img, rowvar=False)
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # 选择最大的n_components个特征向量
    top_eigenvectors = eigenvectors[:, :n_components]
    
    # 将数据映射到低维空间
    transformed_data = np.dot(normalized_img, top_eigenvectors)
    
    # 将低维数据重新映射回原始高维空间
    reconstructed_data = np.dot(transformed_data, top_eigenvectors.T)
    
    # 将重构的数据重新转换为图像矩阵
    reconstructed_img = np.reshape(reconstructed_data, gray_image.shape)
    
    # 绘制原始图像和重构图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title('Reconstructed Image')
    plt.show()

# 示例用法
image = cv2.imread('image.jpg')

# 应用PCA
n_components = 50
pca_image(image, n_components)
