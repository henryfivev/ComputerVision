# SYSU 模式识别与计算机视觉

中山大学李冠彬老师 2023春

## 一、概述



## 二、像素与滤波器



## 三、边缘检测

### Sobel过滤器



### Canny过滤器



### LoG算法

Laplacian of Gaussian

## 四、特征检测

### 1、直线检测

#### 霍夫变换

Hough transform

#### RANSAC

（直线，园。。。）

随机选两个点，确定一条直线，看看有多少点落在该直线上

#### 不变性



### 2、角点检测

#### Moravec

略

#### Harris角点检测器

##### 原理

窗口在角点上按任意角度移动时，窗口的灰度图都会有明显的变化。

窗口滑动分别按x和y方向移动[u, v]后，灰度的变化为

$$
E(u,v) = \sum_{x,y}w(x,y) { [I(x+u,y+v)-I(x,y)]^2}
$$
${w(x,y)}$是窗口函数，二维的滤波器。

${I(x+u,y+v)}$和${I(x,y)}$分别是平移前和平移后的窗口灰度图。

再用泰勒公式${f(x+u,y+v) = f(x,y)+uf_x(x,y)+vf_y(x,y)}$ 简化为

$$
E(x,y)=[u,v]M{ \left[
 \begin{matrix}
   u\\v
  \end{matrix}
  \right]}
$$

$$
M=\sum_{x,y}w(x,y){ \left[
 \begin{matrix}
   I_x^2&I_xI_y\\
   I_xI_y&I_y^2
  \end{matrix}
  \right]}
$$

${I_x}$和${I_y}$为x和y方向的梯度值，可以用Sobel进行计算。

接着${M}$可以用实对称矩阵对角化进一步化简：

$$
M=R^{-1}
{\left[
 \begin{matrix}
   \lambda_1&0\\
   0&\lambda_2
  \end{matrix}
  \right] }
  R
$$
最后根据${\lambda_1}$和${\lambda_2}$计算角点响应函数R：

$$
R={\lambda_1\lambda_2}-k(\lambda_1+\lambda_2)^2
$$
k为经验常数，一般取0.04-0.06。

当R很小且小于threshold时，认为是平坦区域；

当R<0且R<threshold时，认为是边缘；

当R>0且R>threshold时，认为是角点。

##### 步骤

1. 计算${I_x}$和${I_y}$
2. 计算${I_xI_y}$
3. 把${w(x,y)}$，一般是高斯滤波器，应用到${I_x}$、${I_y}$和${I_xI_y}$上，计算$M$矩阵
4. 计算响应值R

##### 特性

平移不变性

旋转不变性

不满足尺度不变性

#### 高斯差分滤波器（DoG）

尺度不变性

## 五、特征描述子

### SIFT描述子



### HOG描述子



## 六、图像缩放

### Seam Carving

能量图

算法优化

### 应用

缩小&放大

去除物体

视频中应用

## 七、图像分割与聚类

超像素

### Agglomerate Clustering

层次聚类的一种

Gestalt理论

### K-means



### Mean-shift Clustering



## 八、降维



## 九、人脸识别



## 十、视觉词袋
