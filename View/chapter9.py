import cv2
import numpy as np
def empty():
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



path = "D:/opencv/Resource/lambo.png"
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,150)


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",152,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",143,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",254,255,empty)
cv2.createTrackbar("Val Min","TrackBars",152,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

def findcolor(img):

    cv2.imshow("Vedio ",img)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hmin = cv2.getTrackbarPos("Hue Min", "TrackBars")
    hmax = cv2.getTrackbarPos("Hue Max", "TrackBars")
    smin = cv2.getTrackbarPos("Sat Min", "TrackBars")
    smax = cv2.getTrackbarPos("Sat Max", "TrackBars")
    vmin = cv2.getTrackbarPos("Val Min", "TrackBars")
    vmax = cv2.getTrackbarPos("Val Max", "TrackBars")
    #print(hmin,hmax,smin,smax,vmin,vmax)
    lower = np.array([hmin,smin,vmin])
    upper = np.array([hmax,smax,vmax])
    print(lower,upper)
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    # cv2.imshow("HSV", imgHSV)
    # cv2.imshow("Img", img)
    # cv2.imshow("mask", mask)
    # cv2.imshow("img Result", imgResult)

    imgStack = stackImages(0.6,([img,imgHSV],[mask,imgResult]))
    cv2.imshow("img Stack",imgStack)
    cv2.waitKey(1)

while True:
    success,img = cap.read()
    findcolor(img)
    #cv2.imshow("Vedio",img)
    if (cv2.waitKey(1) & 0xFF==ord('q')):
        break
