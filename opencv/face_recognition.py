import os
import cv2


face_xml = os.path.expanduser(
    '~/dev/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_xml = os.path.expanduser(
    '~/dev/opencv/data/haarcascades/haarcascade_eye.xml')


face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)


def load_video(fname):
    cap = cv2.VideoCapture(fname)
    ret, frame = cap.read()

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        process_image(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        new_frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        frame = new_frame or frame

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', frame)


frame = load_video('out_5.avi')
# frame = cv2.imread('fac2.jpg')
# cv2.waitKey(0)
cv2.destroyAllWindows()
