import numpy as np
import cv2 as cv

# 读图片
img = cv.imread('task.png',0)

#变大小
# 设置缩放比例
scale_percent = 35  # 缩放比例，这里是35%

# 计算缩放后的新尺寸
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
new_size = (width, height)

# 缩放图片
resized_image = cv.resize(img, new_size)
origin_img = resized_image

#霍夫普圈识别圆圈
img = cv.medianBlur(resized_image,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=105,maxRadius=130)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # 绘制外圆
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # 绘制圆心
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

# 新的窗口提取目标图片
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

# 虚化图片      
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(contour_image,kernel,iterations = 10)

# img = cv.resize(erosion,(0,0),None,fx=1,fy=1)
# img = cv.medianBlur(img,5)
# cimg = cv.cvtColor(img,cv.COLOR_BGR2RGB)
# circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         # 绘制外圆
#         cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         cv.circle(origin_img, (i[0], i[1]), i[2], (255, 0, 0), 2)
#         # 绘制圆心
#         cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

ret, thresh = cv.threshold(erosion, 120, 255, 0) #应用二值化阈值灰度图片

# 寻找轮廓
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

area_threshold = 50 # 设置阈值，用于排除小于该面积的轮廓

for contour in contours:
    area = cv.contourArea(contour)
    if area > area_threshold:
        # 获取轮廓的中心坐标
        M = cv.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # 在图像上画出轮廓
        cv.drawContours(erosion, contour, -1, (0, 255, 0), 2)
        
        # 在指定坐标上输入文字
        text1 = "Volleyball"
        text2 = "(%d,%d)" %(cX,cY)
        cv.putText(origin_img, text1, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv.putText(origin_img, text2, (cX, cY+50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        

cv.imshow('detected circles',cimg)
cv.imshow('contour_image',contour_image)
cv.imshow('thresh',thresh)
cv.imshow('erosion',origin_img)

cv.waitKey(0)
cv.destroyAllWindows()