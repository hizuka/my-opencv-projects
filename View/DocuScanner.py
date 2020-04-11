import cv2
import numpy as np
cap = cv2.VideoCapture(0)
fragmentWidth  =640
fragmentheight = 480
cap.set(3,fragmentWidth)
cap.set(4,fragmentheight)
cap.set(10,150)

def getContours(img):
    contours,herarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)

        if area>500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)

def preprocessing(img):
    imggrav = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imggrav,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel=kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)

    return imgThres





while True:
    success,img = cap.read()
    cv2.resize(img,(fragmentWidth,fragmentheight))
    imgCountour = img.copy()
    imgTres = preprocessing(img)
    cv2.imshow("Vedio",imgTres)
    if (cv2.waitKey(1) & 0xFF==ord('q')):
        break