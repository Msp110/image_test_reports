import cv2
import numpy as np

def estimate_noise(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = np.abs(gray_image.astype(np.float32) - cv2.GaussianBlur(gray_image, (0, 0), 3))
    return np.mean(noise)

def wiener_filter(image, noise_var):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    restored_image = cv2.fastNlMeansDenoising(gray_image, None, h=10.0, templateWindowSize=7, searchWindowSize=21)
    noise_estimation = noise_var / np.var(gray_image)
    wiener_image = cv2.normalize(restored_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return wiener_image

# 读取带噪声的图像
image_path = "D://lena.jpg"
noisy_image = cv2.imread(image_path)

# 估计噪声方差
noise_variance = estimate_noise(noisy_image)

# 使用维纳滤波恢复图像
restored_image = wiener_filter(noisy_image, noise_variance)

# 显示原始图像、带噪声图像和恢复后的图像
cv2.imshow("Original Image", noisy_image)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Restored Image", restored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
