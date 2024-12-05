import numpy as np
from scipy.spatial import cKDTree
import cv2
from skimage.metrics import structural_similarity as ssim
import os

brush_size = 2  # 初始画笔粗细
drawing = False  # 是否在绘制
points = []  # 记录用户画过的像素点


class RBFInterpolator:
    def __init__(self, location, data, epsilon=1, kernel='gaussian', neighbors=10):
        self.location = np.array(location)
        self.data = np.array(data)
        self.epsilon = epsilon
        self.kernel = kernel
        self.neighbors = neighbors
        self.tree = cKDTree(self.location)  # 创建KD树以加速邻居查询

    def _rbf_function(self, r):
        if self.kernel == 'gaussian':
            return np.exp(-(self.epsilon * r) ** 2)
        elif self.kernel == 'multiquadric':
            return np.sqrt((self.epsilon * r) ** 2 + 1)
        elif self.kernel == 'inverse_multiquadric':
            return 1.0 / np.sqrt((self.epsilon * r) ** 2 + 1)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

    def _compute_weights(self, distances, indices):
        n = len(indices)
        r_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                r_matrix[i, j] = self._rbf_function(distances[i][j])
        weights = np.linalg.solve(r_matrix, self.data[indices])
        return weights

    def __call__(self, x, y=None):
        if y is not None:
            # 支持 2D 插值
            points = np.column_stack((x, y))
        else:
            # 支持 1D 插值
            points = np.array(x).reshape(-1, 1)
        distances, indices = self.tree.query(points, k=self.neighbors)  # 找到最近邻
        zi = np.zeros(len(points))
        for i in range(len(points)):
            local_distances = distances[i]
            local_indices = indices[i]
            weights = self._compute_weights(local_distances, local_indices)
            # 计算插值值
            for j in range(self.neighbors):
                r = local_distances[j]
                zi[i] += weights[j] * self._rbf_function(r)
        return zi

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def draw_line(event, x, y, flags, param):
    global points, brush_size, drawing 
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))  # 记录当前画笔的位置
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 在画布上画线，使用当前画笔大小
            cv2.line(mask, points[-1], (x, y), (0,0,0), brush_size)  # 黑色的线条
            points.append((x, y))  # 记录当前画笔的位置
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(mask, points[-1], (x, y), (0,0,0), brush_size)  # 完成绘制

def generate_mask(width,height):
    global mask, brush_size
    mask = np.ones((width, height, 3), dtype=np.uint8) * 255
    cv2.namedWindow('Mask')
    cv2.setMouseCallback('Mask', draw_line)
    # 生成一个mask_size大小的空白图片
    # 画布大小
    while True:
        # 显示图片
        cv2.imshow('Mask', mask)
        # 监听按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 'q' 退出
            break
        elif key == ord('+'):  # 按 '+' 增大画笔
            brush_size += 1
        elif key == ord('-'):  # 按 '-' 减小画笔
            if brush_size > 1:
                brush_size -= 1
    # 关闭窗口
    cv2.destroyAllWindows()
    # 将img保存到文件
    cv2.imwrite('./mask.png', mask)
    return mask

def get_points(mask):
    # 获取mask中所有非零像素的坐标
    points = []
    height, width, channels = mask.shape  # 注意这里包括了通道数
    for y in range(height):
        for x in range(width):
            if mask[y][x][0]==0 and mask[y][x][1]==0 and mask[y][x][2]==0:
                points.append((x, y))  # 记录这个像素的坐标
    return points

def save_masked_img(img, mask, points):
    interpolated_img = img.copy()
    for x,y in points:
        interpolated_img[y][x] = 0  # 黑色像素
    cv2.imwrite('./masked_img.png', interpolated_img)
    return interpolated_img

def nearest_neighbor_interpolation(img, points):
    interpolated_img = img.copy()
    for x,y in points:
        # 寻找欧氏距离最近的不在mask中的像素
        min_distance = float('inf')
        for i in range(x-brush_size, x+brush_size+1):
            for j in range(y-brush_size, y+brush_size+1):
                if (i,j) not in points:  # 去掉不需要插值的像素
                    distance = (x-i)**2 + (y-j)**2  # 计算欧氏距离
                    if distance < min_distance:
                        min_distance = distance
                        interpolated_img[y][x] = img[j][i]  # 用最近的像素填充当前像素
    cv2.imwrite('./NNI.png', interpolated_img)

def bilinear_interpolation(img,points):
    interpolated_img = img.copy()
    for x,y in points:
            left=x-1
            right=x+1
            top=y-1
            bottom=y+1
            while left>=0 and (left,y) in points:
                left-=1
            while right<img.shape[1] and (right,y) in points:
                right+=1
            while top>=0 and (x,top) in points:
                top-=1
            while bottom<img.shape[0] and (x,bottom) in points:
                bottom+=1
            #创建变量计算像素的三通道相似度
            similariy_vertical = ((img[top][y][0]-img[bottom][y][0])**2+(img[top][y][1]-img[bottom][y][1])**2+(img[top][y][2]-img[bottom][y][2])**2)**0.5
            similariy_horizontal = ((img[y][left][0]-img[y][right][0])**2+(img[y][left][1]-img[y][right][1])**2+(img[y][left][2]-img[y][right][2])**2)**0.5
            if(right - left)*similariy_horizontal < (top - bottom)*similariy_vertical:
            # 对(x,y)使用left-right对线性插值
                for i in range (0,3):
                    interpolated_img[y][x][i] = (img[y][left][i]*(right-x) + img[y][right][i]*(x-left))/(right-left)
            else:
                for i in range (0,3):
                    interpolated_img[y][x][i] = (img[top][x][i]*(bottom-y) + img[bottom][x][i]*(y-top))/(bottom-top)
    cv2.imwrite('./BNI.png', interpolated_img)


def rbf_interpolation(img, mask):
    # nearest_neighbor_interpolation(img, mask)
    location = np.transpose(np.nonzero(mask))
    x = np.transpose(np.where(mask == 0))
    data = img[np.nonzero(mask)]

    y_RBF_GAUSSIAN = RBFInterpolator(location, data, epsilon=1, kernel='gaussian', neighbors=12)(x)
    ans_RBF_GAUSSIAN = np.zeros(img.shape)
    ans_RBF_GAUSSIAN[np.nonzero(mask)] = data
    ans_RBF_GAUSSIAN[np.where(mask == 0)] = y_RBF_GAUSSIAN
    cv2.imwrite('./RBF.png', ans_RBF_GAUSSIAN)

# 在大小为width和height的图片上随机生成缺失像素点，比率为ratio
def generate_missing_points(width, height, ratio):
    num_missing = int(width * height * ratio)
    missing_points = []
    while len(missing_points) < num_missing:
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        if (x, y) not in missing_points:
            missing_points.append((x, y))
    mask = np.zeros((width, height))
    for x, y in missing_points:
        mask[y][x] = 1
    return mask


def MSE(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse


def evaluate():
    score_NNI = []
    score_BNI = []
    score_RBF_TPS = []
    score_RBF_GAUSSIAN = []
    diff_NNI = []
    diff_BNI = []
    diff_RBF_TPS = []
    diff_RBF_GAUSSIAN = []
    img_path_prefix = './Ans/example/Randomly remove '
    for i in range (1,10):
        img_path = img_path_prefix + str(i) + '0%/'

        original_img = cv2.imread(os.path.join(img_path, 'Orignal.bmp'))
        NNI = cv2.imread(os.path.join(img_path, 'NNI.bmp'))
        BNI = cv2.imread(os.path.join(img_path, 'BNI.bmp'))
        RBF_TPS = cv2.imread(os.path.join(img_path, 'RBF_TPS.bmp'))
        RBF_GAUSSIAN = cv2.imread(os.path.join(img_path, 'RBF_GAUSSIAN.bmp'))
        # print(ssim(NNI, original_img, multichannel=True, full=True,channel_axis=2)[1])
        score_NNI.append(ssim(NNI, original_img, multichannel=True, full=True,channel_axis=2)[0])
        score_BNI.append(ssim(BNI, original_img, multichannel=True, full=True,channel_axis=2)[0])
        score_RBF_TPS.append(ssim(RBF_TPS, original_img, multichannel=True, full=True,channel_axis=2)[0])
        score_RBF_GAUSSIAN.append(ssim(RBF_GAUSSIAN, original_img, multichannel=True, full=True,channel_axis=2)[0])
        diff_NNI.append(MSE(NNI, original_img))
        diff_BNI.append(MSE(BNI, original_img))
        diff_RBF_TPS.append(MSE(RBF_TPS, original_img))
        diff_RBF_GAUSSIAN.append(MSE(RBF_GAUSSIAN, original_img))

    # 绘制图像，横轴为随机丢失像素的百分比，纵轴为四种算法的SSIM值，百分比依次为10%、20%、30%、40%、50%、60%、70%、80%、90%
    import matplotlib.pyplot as plt
    # plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90], score_NNI, label='NNI', marker='o',linestyle='--')
    # plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90], score_BNI, label='BNI', marker='x',linestyle='-.')
    # plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90], score_RBF_TPS, label='RBF_TPS', marker='s',linestyle='-')
    # plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90], score_RBF_GAUSSIAN, label='RBF_GAUSSIAN', marker='*',linestyle=':')
    # plt.xlabel('Ratio of missing pixels')
    # plt.ylabel('SSIM')
    # plt.title('SSIM of different interpolation methods')
    # plt.grid(linestyle='--')
    # plt.legend()
    # plt.show()

    # 绘制图像，横轴为随机丢失像素的百分比，纵轴为四种算法的MSE值，百分比依次为10%、20%、30%、40%、50%、60%、70%、80%、90%
    plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90], diff_NNI, label='NNI', marker='o',linestyle='--')
    plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90], diff_BNI, label='BNI', marker='x',linestyle='-.')
    plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90], diff_RBF_TPS, label='RBF_TPS', marker='s',linestyle='-')
    plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90], diff_RBF_GAUSSIAN, label='RBF_GAUSSIAN', marker='*',linestyle=':')
    plt.xlabel('Ratio of missing pixels')
    plt.ylabel('L2')
    plt.title('L2 of different interpolation methods')
    plt.grid(linestyle='--')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    evaluate()