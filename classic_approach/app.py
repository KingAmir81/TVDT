import cv2 as cv
import numpy as np
import time
from logger import Logger
if __name__ == "__main__":
    backSub = cv.createBackgroundSubtractorKNN()

    cap = cv.VideoCapture('./src/2103099-hd_1280_720_60fps.mp4')
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv.GaussianBlur(frame,(7,7),5)
        fgmask = backSub.apply(frame)
        _ , fgmask = cv.threshold(fgmask,200,255,cv.THRESH_BINARY)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Find contours
        contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter small contours
            if cv.contourArea(contour) > 500:
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        end = time.time()
        # Display frame
        cv.imshow("frame",fgmask)
        print(1/(end-start))
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
