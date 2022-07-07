# importing the libraries
import cv2
from fer import FER
# Creating FER object
detector=FER()
cap = cv2.VideoCapture(0)
while True:
    # reading the video frames
    result,image=cap.read()
    # resizing the frame
    image = cv2.resize(image, (0, 0), fx = 0.8, fy = 0.8)

    try:
        # using detect_emotions function to get bounding boxes
        # and confidence values of the various emotions
        results=detector.detect_emotions(image)
        print(results)
        # acessing the values to draw bounding boxes
        x,y,w,h=results[0]["box"]
        # top_emotion function to get the maximum confidence value
        emotion,score=detector.top_emotion(image)
        # drawing rectangle on the image
        cv2.rectangle(image,(x,y),(x+w,y+h),(255, 0, 0),2)
        font = cv2.FONT_HERSHEY_PLAIN
        # writing Emotion text on the image
        cv2.putText(image, str(emotion), (x+w-70, y+h+30), font, 2, (0,0,255), 2)
    except IndexError as e:
        pass

    cv2.imshow("image",image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()