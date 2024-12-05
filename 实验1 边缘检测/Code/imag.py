import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# 读取五张图片
img1 = mpimg.imread('./image/3.jpg', cv2.IMREAD_GRAYSCALE)
img2 = mpimg.imread('./image/3_sobel.jpg', cv2.IMREAD_GRAYSCALE)
img3 = mpimg.imread('./image/3_prewitt.jpg', cv2.IMREAD_GRAYSCALE)
img4 = mpimg.imread('./image/3_canny.jpg', cv2.IMREAD_GRAYSCALE)
img5 = mpimg.imread('./image/3_fDoG.jpg', cv2.IMREAD_GRAYSCALE)

# 创建一个包含5个子图的图形
fig, axs = plt.subplots(1, 5, figsize=(15, 5))

# 将每张图片放入对应的子图中
axs[0].imshow(img1, cmap='gray')
axs[0].axis('off')  # 关闭坐标轴
axs[0].set_title('original image')

axs[1].imshow(img2, cmap='gray')
axs[1].axis('off')
axs[1].set_title('Sobel edge detection')

axs[2].imshow(img3, cmap='gray')
axs[2].axis('off')
axs[2].set_title('Prewitt edge detection')

axs[3].imshow(img4, cmap='gray')
axs[3].axis('off')
axs[3].set_title('Canny edge detection')

axs[4].imshow(img5, cmap='gray')
axs[4].axis('off')
axs[4].set_title('fDoG edge detection')

# 展示图形
plt.tight_layout()
plt.show()
