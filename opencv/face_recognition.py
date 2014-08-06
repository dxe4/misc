import os
import cv2


face_xml = os.path.expanduser(
    '~/dev/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_xml = os.path.expanduser(
    '~/dev/opencv/data/haarcascades/haarcascade_eye.xml')


face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)


def get_frame(fname, num):
    cap = cv2.VideoCapture(fname)
    ret, frame = cap.read()
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            raise IndexError('Max: {}'.format(i))

        if i == num:
            return frame

        i += 1


fname = 'out_4.avi'
frame = get_frame(fname, 120)
# frame = cv2.imread('fac2.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray)
for (x, y, w, h) in faces:
    # FIXME
    # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    print(eyes)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('img', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
