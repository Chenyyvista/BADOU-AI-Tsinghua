import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


def generate_gaussian_kernel(kernel_size, sigma):
    #   https://blog.csdn.net/m0_38007695/article/details/82806916
    # 生成高斯核
    W, H = kernel_size
    r, c = np.mgrid[0:H:1, 0:W:1]
    r -= (H - 1) // 2
    c -= (W - 1) // 2

    gaussian_matrix = np.exp(
        -2 * math.pow(sigma, 2) * (np.multiply(r, r) + np.multiply(c, c)) / (2 * math.pi * math.pow(sigma, 2)))
    # 计算高斯矩阵的和
    sunGM = np.sum(gaussian_matrix)
    # 归一化
    gauss_kernel = gaussian_matrix / sunGM
    return gauss_kernel


def pad_zero(img, kernel):
    width, height = img.shape[:2]
    kw, kh = kernel.shape[:2]
    kw = kw // 2
    kh = kh // 2
    new_h = height + kh * 2
    new_w = width + kw * 2

    img_pad = np.zeros((new_h, new_w), np.uint8)
    img_pad[kh:new_h - kh, kw:new_w - kw] = img
    return img_pad


def conv(img, kernel):
    width, height = img.shape[:2]
    kw, kh = kernel.shape[:2]

    img_dst = np.zeros((height - kh + 1, width - kw + 1))
    dst_w, dst_h = img_dst.shape[:2]

    for h in range(dst_h):
        for w in range(dst_w):
            temp = img[h:h + kh, w:w + kw]
            img_dst[h, w] = np.sum(temp * kernel)

    return img_dst


def conv_fast(img, kernel):
    width, height = img.shape[:2]
    kw, kh = kernel.shape[:2]

    dst_h, dst_w = height - kh + 1, width - kw + 1

    img_buffer = []
    for h in range(kh):
        for w in range(kw):
            temp = img[h:h + dst_h, w:w + dst_w]
            temp_1d = np.squeeze(temp.reshape(-1, 1))
            img_buffer.append(temp_1d)
    img_buffer = np.array(img_buffer).T
    kernel_1d = np.array(np.squeeze(kernel.reshape(-1, 1)))

    img_dst = np.dot(img_buffer, kernel_1d)
    img_dst = np.int32(img_dst.reshape(dst_h, dst_w))

    return img_dst


def cal_gradient(img):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_pad = pad_zero(img, sobel_kernel_x)
    gradient_x = conv_fast(img_pad, sobel_kernel_x)
    gradient_y = conv_fast(img_pad, sobel_kernel_y)

    img_gradient = np.sqrt(np.multiply(gradient_x, gradient_x) + np.multiply(gradient_y, gradient_y))
    gradient_x[gradient_x == 0] = 0.00000001
    img_angle = gradient_y / gradient_x
    return img_gradient, img_angle


def non_maximum_suppress(img_gradient, img_angle):
    img_suppress = np.zeros(img_gradient.shape)
    width, height = img_gradient.shape[:2]
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_gradient[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if img_angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / img_angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / img_angle[i, j] + temp[2, 1]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif img_angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / img_angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / img_angle[i, j] + temp[2, 1]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif img_angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * img_angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * img_angle[i, j] + temp[1, 0]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif img_angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * img_angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * img_angle[i, j] + temp[1, 2]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            if flag:
                img_suppress[i, j] = img_gradient[i, j]
    return img_suppress


def gaussian_blur(img, kernel_size, sigma=0.5):
    gauss_kernel = generate_gaussian_kernel(kernel_size, sigma)
    img_new = pad_zero(img, gauss_kernel)
    img_blur = conv_fast(img_new, gauss_kernel)

    return img_blur


def canny(img, kernel_size):
    img_blur = gaussian_blur(img, kernel_size)
    img_gradient, img_angle = cal_gradient(img_blur)
    img_res = non_maximum_suppress(img_gradient, img_angle)

    return img_res.astype(np.uint8)


if __name__ == '__main__':
    pic_path = '../pics/lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值就是灰度化了

    # 高斯平滑
    kernel_size = (5, 5)
    img_canny = canny(img, kernel_size)

    cv2.imshow("canny", img_canny)
    cv2.waitKey(0)
