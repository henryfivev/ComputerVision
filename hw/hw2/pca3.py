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
    print(top_eigenvectors)

    # 将前49个主成分重新转换为图像矩阵
    reconstructed_img = np.reshape(top_eigenvectors.T, (n_components, 32, 32))
    reconstructed_img = (reconstructed_img - np.min(reconstructed_img)) / (np.max(reconstructed_img) - np.min(reconstructed_img)) * 255
    reconstructed_img = reconstructed_img.astype(np.uint8)
    print(reconstructed_img.shape)

    # 前49张人脸可视化
    result = np.zeros([224, 224])

    for i in range(7):
        for j in range(7):
            img = reconstructed_img[i * 7 + j]
            result[i * 32 : i * 32 + 32, j * 32 : j * 32 + 32] = img.T
    result.astype(np.uint8)
    # 绘制原始图像和重构图像
    cv2.imwrite("./result/recovered_faces_top_{}.jpg".format(n_components), result)
    cv2.imshow('Reconstructed Image{}'.format(n_components), result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 这里可视化前49个主成分
    # 读取MAT文件并获取数据，data中有5000张32x32的人脸
    data = loadmat("./data/faces.mat")["X"]

    # 应用PCA
    pca_image(data, 49)
