import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def quantize_image(image, num_bits):
    levels = 2 ** num_bits
    normalized_image = image.astype(np.float32) / 255.0
    quantized_image = np.round(normalized_image * (levels - 1)) / (levels - 1)
    quantized_image = (quantized_image * 255.0).astype(np.uint8)
    return quantized_image


def dpcm_encode(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape
    encoded_image = np.zeros_like(image, dtype=np.int32)
    predicted_value = 0

    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                encoded_image[i, j] = image[i, j]
                predicted_value = image[i, j]
            elif i == 0:
                predicted_value = image[i, j - 1]
                encoded_image[i, j] = int(image[i, j]) - int(predicted_value)
            elif j == 0:
                predicted_value = image[i - 1, j]
                encoded_image[i, j] = int(image[i, j]) - int(predicted_value)
            else:
                predicted_value = (int(image[i - 1, j]) + int(image[i, j - 1])) // 2
                encoded_image[i, j] = int(image[i, j]) - int(predicted_value)

    return encoded_image


def dpcm_decode(encoded_image):
    height, width = encoded_image.shape
    decoded_image = np.zeros_like(encoded_image, dtype=np.uint8)
    predicted_value = 0

    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                predicted_value = encoded_image[i, j]
                decoded_image[i, j] = encoded_image[i, j]
            elif i == 0:
                predicted_value = decoded_image[i, j - 1]
                decoded_image[i, j] = encoded_image[i, j] + predicted_value
            elif j == 0:
                predicted_value = decoded_image[i - 1, j]
                decoded_image[i, j] = encoded_image[i, j] + predicted_value
            else:
                predicted_value = (decoded_image[i - 1, j] + decoded_image[i, j - 1]) // 2
                decoded_image[i, j] = encoded_image[i, j] + predicted_value

    return decoded_image


# 读取图像
image = cv2.imread('D://lena.jpg', 0)

# 执行 DPCM 编码和解码，并计算 PSNR 和 SSIM 值
for num_bits in [1, 2, 4, 8]:
    quantized_image = quantize_image(image, num_bits)
    encoded_image = dpcm_encode(quantized_image)
    decoded_image = dpcm_decode(encoded_image)

    # 将图像重新调整为原始大小，以便与 quantized_image 和 decoded_image 相匹配
    decoded_image = cv2.resize(decoded_image, (image.shape[1], image.shape[0]))

    psnr = peak_signal_noise_ratio(quantized_image, decoded_image)
    ssim = structural_similarity(quantized_image, decoded_image)

    print(f'Quantization: {num_bits} bits')
    print(f'PSNR: {psnr:.2f} dB')
    print(f'SSIM: {ssim:.4f}\n')

    # 显示编码后的图像和解码后的图像
    cv2.imshow(f'Encoded Image (Quantization: {num_bits} bits)', quantized_image)
    cv2.imshow(f'Decoded Image (Quantization: {num_bits} bits)', decoded_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
