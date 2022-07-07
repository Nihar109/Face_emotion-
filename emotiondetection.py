# importing the library
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# importing fer library
from fer import FER
# Creating FER object
detector=FER()
# reading the image
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    #using detect_emotions function to get bounding boxes
    # and confidence values of the various emotions
    results=detector.detect_emotions(img)
    print(results)
    # acessing the values to draw bounding boxes
    x,y,w,h=results[0]["box"]
    emotion,score=detector.top_emotion(img)
     # drawing rectangle on the image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0),2)
    font = cv2.FONT_HERSHEY_PLAIN
    #  writing Emotion text on the image
    cv2.putText(img, str(emotion), (x+w-70, y+h+30), font, 2, (0,0,255), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) != -1:
        cv2.destroyAllWindows()
        break