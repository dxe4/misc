import cv2
from py_opencv import settings


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
    default_file = settings.FACE_CASCADE
    classifier = cv2.CascadeClassifier(default_file)
    color = settings.FACE_BOX_BGR
    thickness = settings.FACE_BOX_THICKNESS

    def find_eyes(self):
        frame = self.drawn_frame or self.frame
        roi_gray = self.gray[self.y:self.y + self.h, self.x:self.x + self.w]
        roi_color = frame[self.y:self.y + self.h, self.x:self.x + self.w]

        eyes = Eye.find(roi_color, roi_gray)
        self.eyes = eyes
        return eyes


class Eye(CascadeClassifierMixIn):
    default_file = settings.EYE_CASCADE
    classifier = cv2.CascadeClassifier(default_file)
    color = settings.EYE_BOX_BGR
    thickness = settings.EYE_BOX_THICKNESS


class Video(object):

    def __init__(self, size_tuple, in_file=0, out_file=None,
                 out_format='XVID'):
        self.size_tuple = size_tuple
        self.out_format = out_format
        self.in_file = in_file
        self.out_file = out_file

        if out_file:
            self.out_writer = self.make_writer()
        else:
            self.out_writer = None

    def make_writer(self):
        if not self.out_file:
            raise ValueError('Object must have out_file set')

        fourcc = cv2.cv.CV_FOURCC(*self.out_format)
        video_writer = cv2.VideoWriter(self.out_file,
                                       fourcc, 10, self.size_tuple)
        return video_writer

    def frame_generator(self):
        '''
        Yields all frames from the video.
        input_file: default=0 ( 0 -> capture from camera)
        If you press q the generator will stop
        TODO: allow passing key listeners
        '''
        cap = cv2.VideoCapture(self.in_file)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            yield frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    def process_image(self, frame):
        faces = Face.find(frame)

        for face in faces:
            face.draw()
            eyes = face.find_eyes()
            for i in eyes:
                i.draw()

        return faces

    def post_process_image(self, frame):
        if self.out_writer:
            self.out_writer.write(frame)

        cv2.imshow('frame', frame)

    def process(self):
        for frame in self.frame_generator():
            self.process_image(frame)
            self.post_process_image(frame)

in_file = 0
out_file = 'out.avi'

video = Video((600, 600), in_file=in_file, out_file=out_file)
video.process()

cv2.destroyAllWindows()
