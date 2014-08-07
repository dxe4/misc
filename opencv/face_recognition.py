import os
import cv2


face_xml = os.path.expanduser(
    '~/dev/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_xml = os.path.expanduser(
    '~/dev/opencv/data/haarcascades/haarcascade_eye.xml')


face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)


def frame_generator(input_file=None):
    cap = cv2.VideoCapture(input_file or 0)
    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        yield frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def process_video(input_file=None, output_file=None):

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    video_writer = cv2.VideoWriter(output_file,
                                   fourcc, 10, (600, 600))
    for frame in frame_generator(input_file=input_file):
        process_image(frame, video_writer)


def process_image(frame, video_writer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        new_frame = cv2.rectangle(
            frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        frame = new_frame or frame

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    video_writer.write(frame)
    cv2.imshow('frame', frame)


process_video(input_file='out_5.avi', output_file='out_processed.avi')
# frame = cv2.imread('fac2.jpg')
# cv2.waitKey(0)
cv2.destroyAllWindows()
# process_image(frame, video_writer)
