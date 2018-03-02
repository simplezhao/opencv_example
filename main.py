import cv2
import numpy as np
import time
#cap = cv2.VideoCapture(1)
img = cv2.imread('test_red2.png')
kernal = np.ones((5,5), np.uint8)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# BGR
while True:
    #ret, frame = cap.read()
    #cv2.line(frame, (0, 0), (640, 480), (255, 0, 0), 5)
    #cv2.rectangle(frame, (384,0),(510,128),(0,255,0),-1)
    #cv2.circle(frame, (447, 63), 63, (0, 0, 255), -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,43,46])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    erosion = cv2.erode(mask, kernal, iterations=1)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contoures, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contoures[0]
    M = cv2.moments(cnt)

    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    cv2.imshow('erosion', erosion)
    #cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break
    if key & 0xff == ord('s'):
        print(frame.shape)
        print(M)
        print('save image')
        #filename = str(int(time.time())) +'.png'
        #cv2.imwrite(filename, frame)

#cap.release()
cv2.destroyAllWindows()


