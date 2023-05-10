import cv2
import numpy as np

def pca_image(data_raw, n_components):
    # 将rgb图像转换成(h*w, 3)
    data = data_raw.reshape(-1, 3)

    # 数据归一化
    normalized_img = (data - np.mean(data)) / np.std(data)

    # 计算协方差矩阵
    covariance_matrix = np.cov(normalized_img, rowvar=False)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print(eigenvectors.shape)

    # 选择最大的n_components个特征向量
    top_eigenvectors = eigenvectors[:, :n_components]
    print(top_eigenvectors.shape)

    # 将数据映射到低维空间
    transformed_data = np.dot(normalized_img, top_eigenvectors)

    # 将低维数据重新映射回原始高维空间
    reconstructed_data = np.dot(transformed_data, top_eigenvectors.T)

    # 将重构的数据重新转换为图像矩阵
    reconstructed_img = np.reshape(reconstructed_data, data_raw.shape)
    reconstructed_img = (reconstructed_img - np.min(reconstructed_img)) / (np.max(reconstructed_img) - np.min(reconstructed_img)) * 255
    reconstructed_img = reconstructed_img.astype(np.uint8)
    print(np.mean(reconstructed_img))
    print(np.max(reconstructed_img))
    print(np.min(reconstructed_img))
    

    # 绘制原始图像和重构图像
    cv2.imwrite("./result/recovered_lena_top_{}.jpg".format(n_components), reconstructed_img)
    cv2.imshow('Reconstructed Image{}'.format(n_components), reconstructed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread("./data/lena.jpg")
    print(img.shape)

    # 应用PCA
    n_components = 10
    pca_image(img, n_components)
    n_components = 50
    pca_image(img, n_components)
    n_components = 100
    pca_image(img, n_components)
    n_components = 150
    pca_image(img, n_components)
