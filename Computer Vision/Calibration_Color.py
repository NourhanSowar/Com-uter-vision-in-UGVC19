import numpy as np
import cv2 
#import serial
import time



capture = cv2.VideoCapture(0)

#img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')

def nothing(x):
    pass

cv2.createTrackbar('B_low' ,'image',0,255,nothing)
cv2.createTrackbar('B_high','image',0,255,nothing)
cv2.createTrackbar('R_low' ,'image',0,255,nothing)
cv2.createTrackbar('R_high','image',0,255,nothing)
cv2.createTrackbar('G_low' ,'image',0,255,nothing)
cv2.createTrackbar('G_high','image',0,255,nothing)


while True :
    
    ret, frame = capture.read()

    r_low = cv2.getTrackbarPos('R_low','image')
    g_low = cv2.getTrackbarPos('G_low','image')
    b_low = cv2.getTrackbarPos('B_low','image')
    r_high = cv2.getTrackbarPos('R_high','image')
    g_high = cv2.getTrackbarPos('G_high','image')
    b_high = cv2.getTrackbarPos('B_high','image')

    lower = np.array([b_low, r_low , g_low] , 'uint8')
    upper = np.array([b_high, r_high, g_high] , 'uint8')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

         

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #mask = cv2.erode(mask, None, iterations=1)
    #mask = cv2.dilate(mask, None, iterations=1)
    #mask = cv2.erode(mask, None, iterations=1)
    blurred = cv2.GaussianBlur(mask, (11, 11), 0)


    output = cv2.bitwise_and(frame, frame , mask = blurred)
     
    contour = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        
    if len(contour) > 0:
        c = max(contour, key = cv2.contourArea )
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
    
    #cv2.imshow('yellow ball detection 1 ',output)
    cv2.imshow('yellow ball detection ',mask)
    #cv2.imshow('yellow ',output)
    cv2.imshow("images", np.hstack([frame, output]))
    #cv2.imshow("images1", np.hstack([frame1, output1]))

    
    k = cv2.waitKey(1)
    if k==27 :
        break

capture.release()
cv2.destroyAllWindows()
