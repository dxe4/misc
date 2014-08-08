import os
import cv2


settings = {
    # TODO this needs refactoring
    'FACE_CASCADE': os.path.expanduser(
        '~/dev/opencv/data/haarcascades/haarcascade_frontalface_default.xml'),
    'FACE_BOX_BGR': (255, 0, 0),
    'FACE_BOX_THICKNESS': 2,

    'EYE_CASCADE': os.path.expanduser(
        '~/dev/opencv/data/haarcascades/haarcascade_eye.xml'),
    'EYE_BOX_BGR': (0, 255, 0),
    'EYE_BOX_THICKNESS': 2,
}


def fix_eye():
    '''
    If we have 3 eyes something went wrong.
    -> If 2 eyes overlap 100/100 pick the biggest
    -> else If previous frames available:
        -> find eyes in previous frames
        -> If the previous frames have no eyes or have 3 eyes mark face as
           'error' (can draw red)
        -> else if the last 2-3 frames have 2 eyes:
          -> If 1 eye is irrelevant to previous eyes
             and 2 of them are close to previous eyes by 10/100
             or 2 of them overlap 90/100
             discard the other eye
          -> else mark as `error`
    '''
    pass


def eye_ellipse_detection():
    # Ellipse:
    # (x/a)**2 + (y/b)**2 == 1
    # y = +- sqrt((a**2 * b**2 - b**2 * x**2) / a**2)
    """
    max_white =  (0, 0, 0)
    min_white =  (30, 30, 30) where sum(min_white) <= 30
    a) find ellipse in range (max_white - min_white)
    b) find circle on a black & white inversed image
    c) pick a,b from circle
    d) draw oval
    """
    pass


def fix_face():
    '''
    If a face.eyes == 0 or face.eyes > 2 face.error = true
    (unless eyes can be fixed)
    '''
    pass


class CascadeClassifierMixIn(object):

    def __init__(self, rectangle_tuple, frame, gray):
        x, y, w, h = rectangle_tuple
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame = frame
        self.gray = gray
        self.drawn_frame = None

    @classmethod
    def find(cls, frame, gray=None):
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return [cls(i, frame, gray)
                for i in cls.classifier.detectMultiScale(gray)]

    def draw(self):
        self.drawn_frame = cv2.rectangle(self.frame,
                                         (self.x, self.y),
                                         (self.x + self.w, self.y + self.h),
                                         self.color, self.thickness)


class Face(CascadeClassifierMixIn):
    default_file = settings['FACE_CASCADE']
    classifier = cv2.CascadeClassifier(default_file)
    color = settings['FACE_BOX_BGR']
    thickness = settings['FACE_BOX_THICKNESS']

    def find_eyes(self):
        frame = self.drawn_frame or self.frame
        roi_gray = self.gray[self.y:self.y + self.h, self.x:self.x + self.w]
        roi_color = frame[self.y:self.y + self.h, self.x:self.x + self.w]

        eyes = Eye.find(roi_color, roi_gray)
        self.eyes = eyes
        return eyes


class Eye(CascadeClassifierMixIn):
    default_file = settings['EYE_CASCADE']
    classifier = cv2.CascadeClassifier(default_file)
    color = settings['EYE_BOX_BGR']
    thickness = settings['EYE_BOX_THICKNESS']


class Video(object):

    def __init__(self):
        pass


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
    video_writer = video_config.make_writer() if video_config else None

    for frame in frame_generator(input_file=input_file):
        process_image(frame)
        post_process_image(frame, video_writer=video_writer)


def post_process_image(frame, video_writer=None):
    if video_writer:
        video_writer.write(frame)

    cv2.imshow('frame', frame)


def process_image(frame):
    faces = Face.find(frame)

    for face in faces:
        face.draw()
        eyes = face.find_eyes()
        for i in eyes:
            i.draw()

    return faces


default_video_config = VideoOutputConfig('out_processed.avi', 600, 600)
process_video(input_file='out_5.avi', video_config=default_video_config)
cv2.destroyAllWindows()
