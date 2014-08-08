import cv2
import time


def process_image(vide):
    raise NotImplemented


def read_video(fname):
    cap = cv2.VideoCapture(fname)

    while(cap.isOpened()):
        ret, frame = cap.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def capture_for(seconds):
    started = time.time()
    fname = str(started).split('.')[0]

    capture = cv2.VideoCapture(0)
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    # TODO 10 seconds lead to 6 seconds video :S
    video_writer = cv2.VideoWriter("out_{}.avi".format(fname),
                                   fourcc, seconds, (600, 600))

    # record video
    while(capture.isOpened()):
        if started < time.time() - seconds:
            break

        ret, frame = capture.read()

        if ret is True:
            # frame = cv2.flip(frame, 0)
            # write the flipped frame
            video_writer.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    duration = 10
    # capture_for(duration)
    read_video('out_1407145237.avi')
