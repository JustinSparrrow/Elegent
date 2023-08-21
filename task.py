import cv2 as cv
import numpy as np

img = cv.imread('task.png', 1)  # 读取彩色图像

# 设置缩放比例
scale_percent = 35  # 缩放比例，这里是35%

# 计算缩放后的新尺寸
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
new_size = (width, height)

# 缩放图片
resized_image = cv.resize(img, new_size)

imgray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY) # 转化为灰度图像

ret, thresh = cv.threshold(imgray, 180, 255, 0) #应用二值化阈值灰度图片

cv.imshow('Thresh', thresh) #画完轮廓后的图片

blue_channel = resized_image[:,:,0]
green_channel = resized_image[:,:,1]
red_channel = resized_image[:,:,2]

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #查找轮廓

cv.drawContours(resized_image, contours, -1, (0,255,0), 3)

# 显示图像
cv.imshow("image", resized_image)
cv.imshow("blue",blue_channel)
cv.imshow("green",green_channel)
cv.imshow("red",red_channel)

k = cv.waitKey(0)
if k == 27:         # 等待ESC退出
    cv.destroyAllWindows()
elif k == ord('s'): # 等待关键字，保存和退出
    cv.imwrite('contours_on_black.png', resized_image)
    cv.destroyAllWindows()
