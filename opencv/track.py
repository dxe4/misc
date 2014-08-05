import numpy as np
import cv2

fname = 'out_1407275844.avi'
TRACK_COLOR = (np.array((0., 0., 0.)), np.array((100., 100., 100.)))
R, H, C, W = 250, 90, 400, 125
track_window = (C, R, W, H)

cap = cv2.VideoCapture(fname)
ret, frame = cap.read()

# set up the ROI for tracking
roi = frame[R:R + H, C:C + W]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, *TRACK_COLOR)

roi_hist = cv2.calcHist(
    [hsv_roi], [0], mask, [180], [0, 180])

cv2.normalize(
    roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if ret is True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, W, H = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + W, y + H), 255, 2)

        cv2.imshow('img2', frame)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            pass
            # cv2.imwrite("{}.jpg".format(k), img2)
    else:
        break

cv2.destroyAllWindows()
cap.release()
