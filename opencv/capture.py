import cv2
# find the webcam
capture = cv2.VideoCapture(0)


fourcc = cv2.cv.CV_FOURCC(*'XVID')
video_writer = cv2.VideoWriter("output.avi", fourcc, 25, (600, 600))


# video recorder
# cv2.VideoWriter_fourcc() does not exist
# fourcc = cv2.cv.CV_FOURCC(*'XVID')
# video_writer = cv2.VideoWriter("output.avi", -1, 20, (680, 480))

# record video
while(capture.isOpened()):
    ret, frame = capture.read()
    if ret == True:
        frame = cv2.flip(frame, 0)

        # write the flipped frame
        video_writer.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
# video_writer.release()
cv2.destroyAllWindows()
