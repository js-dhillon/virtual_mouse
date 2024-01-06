import cv2
import time
import HandTrackingModule as htm
import numpy as np
import pyautogui

#Works with linux using xorg
#Doesn't work wuith linux using wayland

wCam, hCam = 680, 480
frameR = 100
smoothening = 7

cap = cv2.VideoCapture(0)
cap.set(4,hCam)
cap.set(3, wCam)


pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.handDectector(maxHands=1)
wScr, hScr = pyautogui.size()
print(wScr, hScr)

while True:
    # Find hand Landmarks
    success, img =cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    # Get tip of index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        #Only Index Finger : Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            #Convert Coordiantes
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            #Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            #Move Mouse
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 165, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY
        #Both Index and Middle Fingers are up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:

            # Step9: Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
        #All fingers up: Scroll Mode
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0  and fingers[4] == 0:
            pyautogui.scroll(-2)
        if fingers[3] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0:
           pyautogui.scroll(2)

    #frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display
    cv2.imshow("Img", img)
    cv2.waitKey(1)




