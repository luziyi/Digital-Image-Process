# SSR
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

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
 
def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])  # 找到数组中最小的非零值
    data = np.where(data == 0, min_nonzero, data)  # 将数组中的零值替换为最小的非零值
    return data  # 返回替换后的数组
 
def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)  # 高斯函数
    img = replaceZeroes(src_img)  # 去除0  
    L_blur = replaceZeroes(L_blur)  # 去除0
 
    dst_Img = cv2.log(img / 255.0)  # 归一化取log
    dst_Lblur = cv2.log(L_blur / 255.0)  # 归一化取log
    dst_IxL = cv2.multiply(dst_Img, dst_Lblur)  # 乘  L(x,y)=S(x,y)*G(x,y)
    log_R = cv2.subtract(dst_Img, dst_IxL)  # 减  log(R(x,y))=log(S(x,y))-log(L(x,y))
 
    dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)  # 放缩到0-255
    log_uint8 = cv2.convertScaleAbs(dst_R)  # 取整
    return log_uint8
 
 
def SSR_image(image):
    size = 3
    b_gray, g_gray, r_gray = cv2.split(image)  # 拆分三个通道
    # 分别对每一个通道进行 SSR
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv2.merge([b_gray, g_gray, r_gray])  # 通道合并。
    return result
 
 
if __name__ == "__main__":
    input_img = cv2.imread('./SSR/low/111.png', cv2.IMREAD_COLOR)  # 读取输入图像
    enhanced_img = SSR_image(input_img)  # 调用 SSR 函数得到增强后的图像
    cv2.imwrite('./SSR/enhanced/111.png', enhanced_img)  # 将增强后的图像保存为 img_2.png

    original_img = cv2.imread('./SSR/high/111.png')  # 读取原始图像
    enhanced_img = cv2.imread('./SSR/enhanced/111.png')  # 读取增强后的图像

    # 转换为灰度图像（PSNR 和 SSIM 通常在灰度图像上计算）
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    # 计算 PSNR 和 SSIM
    psnr_value = calculate_psnr(original_gray, enhanced_gray)
    ssim_value = calculate_ssim(original_gray, enhanced_gray)

    print(f"PSNR: {psnr_value} dB")
    print(f"SSIM: {ssim_value}")

    # 显示低光图像，正常图像以及增强后的图像和 PSNR 和 SSIM 值，显示在一个窗口中 使用plt.show()
    # 使用matplotlib显示图像和计算结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示低光图像  
    axes[0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))  # 转换为RGB
    axes[0].set_title('Low-light Image')
    axes[0].axis('off')

    # 显示原始图像
    axes[1].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))  # 转换为RGB
    axes[1].set_title('Original Image')
    axes[1].axis('off')

    # 显示增强后的图像
    axes[2].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))  # 转换为RGB
    axes[2].set_title('Enhanced Image')
    axes[2].axis('off')

    # 在图像上显示 PSNR 和 SSIM 值
    fig.suptitle(f"PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}", fontsize=14)

    # 显示所有图像
    plt.tight_layout()
    plt.show()