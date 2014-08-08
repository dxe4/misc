import os

FOUND_OBJECT_THICKNESS = 2

FACE_CASCADE = os.path.expanduser(
    '~/dev/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
FACE_BOX_BGR = (255, 0, 0)
FACE_BOX_THICKNESS = FOUND_OBJECT_THICKNESS

EYE_CASCADE = os.path.expanduser(
    '~/dev/opencv/data/haarcascades/haarcascade_eye.xml')
EYE_BOX_BGR = (0, 255, 0)
EYE_BOX_THICKNESS = FOUND_OBJECT_THICKNESS
