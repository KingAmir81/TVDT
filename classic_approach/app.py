import cv2 as cv

if __name__ == "__main__":
    backSub = cv.createBackgroundSubtractorKNN()

    cap = cv.VideoCapture('./src/2103099-hd_1280_720_60fps.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.GaussianBlur(frame,(7,7),5)
        fgmask = backSub.apply(frame)
        _ , fgmask = cv.threshold(fgmask,200,255,cv.THRESH_BINARY)
        cv.imshow("frame",fgmask)

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
