import numpy as np
from scipy.ndimage import filters


def compute_harris_response(im, sigma=3):
    """在一幅灰度图像中，对每个像素计算 Harris 角点检测器响应函数"""
    # 计算导数
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # 计算 Harris 矩阵的分量
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # 计算特征值和迹
    Wdet = Wxx * Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """从一幅 Harris 响应图像中返回角点。min_dist 为分割角点和图像边界的最少像素数目"""
    # 寻找高于阈值的候选角点
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # 得到候选点的坐标
    coords = np.array(harrisim_t.nonzero()).T

    # 以及它们的 Harris 响应值
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # 对候选点按照 Harris 响应值进行排序
    index = np.argsort(candidate_values)

    # 将可行点的位置保存到数组中
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # 按照最小距离原则，选择最佳 Harris 点
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[
                (coords[i, 0] - min_dist) : (coords[i, 0] + min_dist),
                (coords[i, 1] - min_dist) : (coords[i, 1] + min_dist),
            ] = 0

    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """绘制图像中检测到的角点"""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], "*")
    plt.axis("off")
    plt.show()
