import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 计算 PSNR
def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)  # 均方误差
    if mse == 0:
        return 100  # 没有差异，PSNR为100
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)  # PSNR 公式
    return psnr

# 计算 SSIM
def calculate_ssim(original, enhanced):
    # 使用 scikit-image 库中的 ssim 函数
    ssim_index, _ = ssim(original, enhanced, full=True)
    return ssim_index

if __name__ == "__main__":
    original_img = cv2.imread('./high-3.png')  # 读取原始图像
    enhanced_img = cv2.imread('./enhanced-3.jpg')  # 读取增强后的图像

    # 转换为灰度图像（PSNR 和 SSIM 通常在灰度图像上计算）
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    # 计算 PSNR 和 SSIM
    psnr_value = calculate_psnr(original_gray, enhanced_gray)
    ssim_value = calculate_ssim(original_gray, enhanced_gray)

    print(f"PSNR: {psnr_value} dB")
    print(f"SSIM: {ssim_value}")
