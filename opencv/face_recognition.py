import os
import cv2


face_xml = os.path.expanduser(
    '~/dev/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_xml = os.path.expanduser(
    '~/dev/opencv/data/haarcascades/haarcascade_eye.xml')


face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)


class VideoOutputConfig(object):

    def __init__(self, file_name, size_x, size_y, video_format='XVID'):
        self.file_name = file_name
        self.size_x = size_x
        self.size_y = size_y
        self.video_format = video_format
        self.size_tuple = (self.size_x, self.size_y)

    def make_writer(self):
        fourcc = cv2.cv.CV_FOURCC(*self.video_format)
        video_writer = cv2.VideoWriter(self.file_name,
                                       fourcc, 10, self.size_tuple)
        return video_writer


def frame_generator(input_file=0):
    '''
    Yields all frames from the video.
    input_file: default=0 ( 0 -> capture from camera)
    If you press q the generator will stop
    TODO: allow passing key listeners
    '''
    cap = cv2.VideoCapture(input_file)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        yield frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def process_video(input_file=None, video_config=None):
    video_writer = video_config.make_writer()

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


default_video_config = VideoOutputConfig('out_processed.avi',
                                         600, 600)
process_video(input_file='out_5.avi', video_config=default_video_config)
# frame = cv2.imread('fac2.jpg')
# cv2.waitKey(0)
cv2.destroyAllWindows()
# process_image(frame, video_writer)
