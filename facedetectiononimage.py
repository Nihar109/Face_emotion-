# importing the libraries
import cv2
# library contains Ml algo
import dlib
# Capturing the video from webcam
cap=cv2.VideoCapture(0)
# dlib.get_frontal_face_detector() is used in detecting the face in a frame or image
detector = dlib.get_frontal_face_detector()
# Creating the predict object
# to detect the keyparts/shapes in the face
while True:
    # Read the frames
    _,frame=cap.read()
    # Converting images into gray
    converting_img_into_grayscale= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecting the faces from the gray image
    # Returns array of Rectangular objects
    faces=detector(converting_img_into_grayscale)
    # face
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        # Drawing  the rectangle on the faces
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imshow("Frame",frame)
    cv2.waitKey(1)
