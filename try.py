import cv2 as cv
import numpy as np

# 读取图像
img = cv.imread('task.png', 0)
img = cv.resize(img, (0, 0), None, fx=0.3, fy=0.3)
img = cv.medianBlur(img, 5)

# 使用霍夫圆变换检测图像中的圆
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=95, maxRadius=120)

# 创建一个新窗口
new_window = np.zeros_like(img)

# 创建一个掩码图像
mask = np.zeros_like(img)

if circles is not None:
    circles = np.uint16(np.around(circles))
    
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]

        # 绘制检测到的圆
        cv.circle(new_window, center, radius, (255, 255, 255), thickness=cv.FILLED)

        cv.circle(mask, center, radius, (255, 255, 255), thickness=cv.FILLED)

        # 使用掩码提取轮廓内的图像
        contour_image = cv.bitwise_and(img, img, mask=mask)

# 显示轮廓内的图像在新窗口上
cv.imshow('Contour Image', contour_image)

# 显示新窗口
cv.imshow('Circles on New Window', new_window)
cv.waitKey(0)
cv.destroyAllWindows()
