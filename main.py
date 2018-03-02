import cv2
import numpy as np
import time
#cap = cv2.VideoCapture(1)
img = cv2.imread('test_red3.png')
kernal = np.ones((5,5), np.uint8)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# BGR
ret,thresh = cv2.threshold(imgray, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(img, contours, 1, (0,255,0), 3)
cnt = contours[0]
M = cv2.moments(cnt)
print(M)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print(cx,cy)
cv2.circle(img,(cx,cy), 2, (255,0,0), -1)
cv2.imshow('img', img)
cv2.waitKey(0)

#cap.release()
cv2.destroyAllWindows()


