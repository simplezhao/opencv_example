import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
test_img = cv2.imread('test_red.png')
kernel = np.ones((5, 5), np.uint8)
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# BGR
# ret,thresh = cv2.threshold(imgray, 127, 255, 0)
# image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,
#                                               cv2.CHAIN_APPROX_SIMPLE)
#
# img = cv2.drawContours(img, contours, 1, (0,255,0), 3)
# cnt = contours[1]
# M = cv2.moments(cnt)
# print(M)
# cx = int(M['m10']/M['m00'])
# cy = int(M['m01']/M['m00'])
# print(cx,cy)
# cv2.circle(img,(cx,cy), 2, (255,0,0), -1)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# 设定蓝色的阈值
lower_red = np.array([0, 43, 46])  # 156
upper_red = np.array([10, 255, 255])


def get_circle(point):
    print('计算圆心：')
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    x3 = 0
    y3 = 0
    x1y1 = 0
    x1y2 = 0
    x2y1 = 0
    point_size = len(point)
    for i in range(point_size):
        x = point[i][0]
        y = point[i][1]
        x1 = x1 + x
        y1 = y1 + y
        x2 = x2 + x * x
        y2 = y2 + y * y
        x3 = x3 + x * x * x
        y3 = y3 + y * y * y
        x1y1 = x1y1 + x * y
        x1y2 = x1y2 + x * y * y
        x2y1 = x2y1 + x * x * y
    N = len(point)
    C = N * x2 - x1 * x1
    D = N * x1y1 - x1 * y1
    E = N * x3 + N * x1y2 - (x2 + y2) * x1
    G = N * y2 - y1 * y1
    H = N * x2y1 + N * y3 - (x2 + y2) * y1
    a = (H * D - E * G) / (C * G - D * D)
    b = (H * C - E * D) / (D * D - G * C)
    c = -(a * x1 + b * y1 + x2 + y2) / N
    A = a / (-2)
    B = b / (-2)
    R = math.sqrt(a * a + b * b - 4 * c) / 2
    print(A, B, R)
    return A, B, R


def get_point(contours, count=0):
    if len(contours) > 0:
        print("计算中心坐标")
        cnt = contours[count]
        M = cv2.moments(cnt)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print('坐标： ', cx, cy)
            return cx, cy
        except:
            print('please retry')
            return None
    else:
        return None


# while True:
ret, frame = cap.read()
gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_red, upper_red)
mask = cv2.erode(mask, kernel, iterations=1)
ret, thresh = cv2.threshold(mask, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)
font = cv2.FONT_HERSHEY_SIMPLEX
# img = cv2.drawContours(frame, contours, 0, (0, 255, 0), 3)
# print(contours)
img = []
point = []
for i in range(5):
    img = cv2.drawContours(frame, contours, i, (0, 255, 0), 3)
    # cv2.bitwise_and()
    tmp = get_point(contours, count=i)
    if tmp is not None:
        point.append(tmp)
        cv2.circle(test_img, tmp, 4, (0, 255, 0), -1)
        text = str(tmp)
        cv2.putText(test_img, text, tmp, font, 0.8, (0, 255, 0), 2)
print(point)

circle = get_circle(point)
#画出拟合的圆
cv2.circle(test_img,(int(circle[0]), int(circle[1])), int(circle[2]), (255, 0, 0), 1)
#画出圆心
cv2.circle(test_img,(int(circle[0]), int(circle[1])), 5, (255, 0, 0), -1)
text = "x,y,R: " + str(int(circle[0])) + "," + str(int(circle[1])) + "," + str(int(circle[2]))
cv2.putText(test_img, text, (25, 25), font, 0.8, (0, 255, 0), 2)
cv2.line(test_img,(100,25),(int(circle[0]), int(circle[1])),(255,0,0),2)

cv2.imshow('test image', test_img)
cv2.imshow('mask', mask)
# cv2.imshow('video', frame)
cv2.waitKey(0)
# if key & 0xff == ord('q'):
#   break

cap.release()
cv2.destroyAllWindows()
