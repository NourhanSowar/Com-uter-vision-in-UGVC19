#### Notes :::::  It is detect what is the length is applied  and then check the colour if it is red or not and then fraw contour to the coloured _detected_object and check the length  >>if 2 lengths is equal  .It is a red cube
### We need only for this code BGR reads of the read colour and a range of length of the real cube.

import cv2
import numpy as np
import time

capture = cv2.VideoCapture(0)
last_time = time.time()


def color ():
    hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)

    lower_red = np.array([97,83 ,45 ])
    upper_red = np.array([120, 223, 154])

    mask = cv2.inRange(hsv, lower_red, upper_red)


    res = cv2.bitwise_and(img, img, mask=mask)
    canny = cv2.Canny(mask, 250, 255)
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    # print(contours )
    if len(contours) > 0:
        cnts = contours[0]
        for contour in contours:
            max_area = cv2.contourArea(contour)
            c = max(contours, key=cv2.contourArea)

            approx_color = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            print( "colored clontour" )
            print(len(approx_color))
            if (5<=len(approx_color) <= 15):
                cv2.putText(img, 'Cube', pt, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, [0, 255, 255], 2)    ##### the actual detection that length is correct and the colour is true.
                print("Cube")







    cv2.imshow('colored_mask', mask)
    cv2.imshow('res',res)

    return ;




while True :

    ret, img = capture.read()


    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.namedWindow("Gray Converted Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Gray Converted Image",img_gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
    noise_removal = cv2.bilateralFilter(img_gray,9,75,75)

    cv2.namedWindow("Noise Removed Image", cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    cv2.imshow("Noise Removed Image", noise_removal)

    ret,thresh_image = cv2.threshold(noise_removal,0,255,cv2.THRESH_OTSU)
    #cv2.namedWindow("Image after Thresholding",cv2.WINDOW_NORMAL)
    cv2.imshow("Image after Thresholding",thresh_image)

# Applying Canny Edge detection
    canny_image = cv2.Canny(thresh_image,250,255)
    #cv2.namedWindow("Image after applying Canny",cv2.WINDOW_NORMAL)
    cv2.imshow("Image after applying Canny",canny_image)
    canny_image = cv2.convertScaleAbs(canny_image)

# dilation to strengthen the edges
    kernel = np.ones((3,3), np.uint8)
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
    #cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
    cv2.imshow("Dilation", dilated_image)


    contours, h = cv2.findContours(dilated_image, 1, 2)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:1]
    pt = (180, 3 * img.shape[0] // 4)
    for cnt in contours:

        c = max(contours, key=cv2.contourArea)

        approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
        #print (len(cnt))
        print  (len(approx))
        if   (5<=len(approx) <=15) :



            cv2.drawContours(img,[cnt],-1,(255,0,0),3)                ###################contour  ll edge detection #########################
            color()
            #cv2.putText(img,'Cube', pt ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, [0,255, 255], 2)


    #cv2.namedWindow("Shape", cv2.WINDOW_NORMAL)
    cv2.imshow('Shape',img)


    corners    = cv2.goodFeaturesToTrack(thresh_image,6,0.06,25)
    corners    = np.float32(corners)
    for    item in    corners:
        x,y    = item[0]
        cv2.circle(img,(x,y),10,255,-1)
    cv2.namedWindow("Corners", cv2.WINDOW_NORMAL)
    cv2.imshow("Corners",img)

    print('Loop took {} seconds '.format(time.time() - last_time))
    last_time = time.time()

    k = cv2.waitKey(1)
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
