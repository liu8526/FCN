import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def Bilinear_interpolation (src, new_size):
    """使用双线性插值方法放大图像
    para:
        src(np.ndarray):输入图像
        new_size:(tuple): 目标尺寸
    ret:
        dst(np.ndarray): 目标图像
    """
    dst_h, dst_w = new_size # 目标图像的高和宽
    src_h, src_w = src.shape[:2]  #源图像的高和宽
    # 尺寸不变则直接返回
    if src_h == dst_h and src_w == dst_w: 
        return src.copy()
    
    scale_x = float(src_w) / dst_w   #缩放比例
    scale_y = float(src_h) / dst_h
    
    #遍历目标图上的每一个像素，由原图的点插入数值
    dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8) #生成一张目标尺寸大小的空白图，遍历插值
    for n in range(3): #循环channel
        for dst_y in range(dst_h): #循环height
            for dst_x in range(dst_w): #循环width
                #目标像素在源图上的坐标
                # src_x + 0.5 = (dst_x + 0.5) * scale_x 
                src_x = (dst_x + 0.5) * scale_x - 0.5  # 一个像素默认为1*1的小格子，其中心在像素坐标点加0.5的位置
                src_y = (dst_y + 0.5) * scale_y - 0.5                
                #计算在源图上四个近邻点的位置
                src_x_0 = int(np.floor(src_x))  #向下取整 floor(1.2) = 1.0
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1, src_w - 1)  #防止出界
                src_y_1 = min(src_y_0 + 1, src_h - 1)                
                #双线性插值 新图像每个像素的值来自于原图像上像素点的组合插值
                value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, n] + \
                         (src_x - src_x_0) * src[src_y_0, src_x_1, n]
                value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, n] + \
                         (src_x - src_x_0) * src[src_y_1, src_x_1, n]
                dst[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + \
                                           (src_y - src_y_0) * value1)         
    return dst

# cv2_img_in = cv2.imread('./CamVid/train/0001TP_006690.png') ##得到256阶bgr结果
# b, g, r = cv2.split(cv2_img_in)
# cv2_img_in = cv2.merge([r, g, b])

# # img_in = plt.imread('./CamVid/train/0001TP_006690.png') #得到归一化rgb结果
# img_out = Bilinear_interpolation(img_in, [img_in.shape[0]*2,img_in.shape[1]*2])

# plt.figure(1)
# plt.subplot(1, 2, 1)
# plt.imshow(img_in) # 显示图片

# plt.subplot(1, 2, 2)
# plt.imshow(img_out)

# 手动设计一个滤子
def bilinear_kernel(in_channels, out_channels, kernel_size):
    """使用双线性插值的方法初始化卷积层中卷积核的权重参数
    para:
        in_channels(int): 输入通道数
        out_channels(int): 输出通道数
        kernel_size()
    ret:
        torch.tensor : a bilinear filter kernel
    """
    factor = (kernel_size + 1) // 2
    center = kernel_size/2
    
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    
    return torch.from_numpy(weight)


# 测试
cv2_img_in_bgr = cv2.imread('./CamVid/train/0001TP_006690.png') #得到256阶bgr结果
cv2_img_in_rgb = cv2.cvtColor(cv2_img_in_bgr, cv2.COLOR_BGR2RGB)
# b, g, r = cv2.split(cv2_img_in_bgr)
# cv2_img_in_rgb = cv2.merge([r, g, b])
# cv2_img_in_rgb = cv2.merge([cv2_img_in_bgr[:,:,2], cv2_img_in_bgr[:,:,1], cv2_img_in_bgr[:,:,0]])
# cv2_img_array = np.array([cv2_img_in_bgr[:,:,2], cv2_img_in_bgr[:,:,1], cv2_img_in_bgr[:,:,0]])#将cv2读取的bgr格式转换成rgb
# img_in_rgb = torch.from_numpy(cv2_img_in_rgb).permute(2, 0, 1).numpy()
img_in_rgb = np.array(cv2_img_in_rgb).astype(np.float)/256#转换成plt读图格式：浮点rgb

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(img_in_rgb)
img_in_tensor = torch.from_numpy(img_in_rgb.astype('float32')).permute(2, 0, 1).unsqueeze(0)  # .unsqueeze(0) 增加 batch_size 通道
conv_trans = nn.ConvTranspose2d(3, 3, 4, 3, 1)
# 将其定义为 bilinear kernel
conv_trans.weight.data = bilinear_kernel(3, 3, 4)
# print(conv_trans.weight)
# print(conv_trans.weight.data)
cv2_img_out_rgb = conv_trans(img_in_tensor).data.squeeze().permute(1, 2, 0).numpy()
plt.subplot(1, 2, 2)
plt.imshow(cv2_img_out_rgb)
print(cv2_img_out_rgb.shape)