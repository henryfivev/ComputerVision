import cv2
import numpy as np
from scipy.io import loadmat


def pca_image(data, n_components):
    # 数据归一化
    normalized_img = (data - np.mean(data)) / np.std(data)

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
    reconstructed_img = np.reshape(reconstructed_data, data.shape)
    reconstructed_img = (reconstructed_img - np.min(reconstructed_img)) / (np.max(reconstructed_img) - np.min(reconstructed_img)) * 255
    reconstructed_img = reconstructed_img.astype(np.uint8)

    # 重建前100张人脸并可视化
    result = np.zeros([320, 320])

    for i in range(10):
        for j in range(10):
            img = np.reshape(reconstructed_img[i * 10 + j, :], (32, 32))
            result[i * 32 : i * 32 + 32, j * 32 : j * 32 + 32] = img.T

    result = result.astype(np.uint8)

    # 绘制原始图像和重构图像
    cv2.imwrite("./result/recovered_faces_top_{}.jpg".format(n_components), result)
    cv2.imshow('Reconstructed Image{}'.format(n_components), result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 这里分别把图像降维到10, 50, 100和150维度后重建并可视化前100张人脸
    # 读取MAT文件并获取数据，data中有5000张32x32的人脸
    data = loadmat("./data/faces.mat")["X"]

    # 应用PCA
    pca_image(data, 10)
    pca_image(data, 50)
    pca_image(data, 100)
    pca_image(data, 150)
    pca_image(data, 1024)
