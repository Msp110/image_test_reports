import cv2
import numpy as np


def threshold_segmentation(image, initial_threshold, max_iterations=100, tolerance=1e-5):
    threshold = initial_threshold
    for i in range(max_iterations):
        # 计算前景和背景的平均灰度值
        foreground_mean = np.mean(image[image > threshold])
        background_mean = np.mean(image[image <= threshold])

        # 更新阈值
        new_threshold = 0.5 * (foreground_mean + background_mean)

        # 判断阈值变化是否小于容差
        if abs(new_threshold - threshold) < tolerance:
            break

        threshold = new_threshold

    # 根据最终的阈值对图像进行分割
    segmented_image = np.zeros_like(image)
    segmented_image[image > threshold] = 255

    return segmented_image, threshold


# 读取图像
image = cv2.imread('D://lena.jpg', 0)  # 灰度图像

# 设置初始阈值和参数
initial_threshold = 128
max_iterations = 100
tolerance = 1e-5

# 进行阈值分割
segmented_image, threshold = threshold_segmentation(image, initial_threshold, max_iterations, tolerance)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
