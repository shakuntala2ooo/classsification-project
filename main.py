import cvzone
import cv2

cap=cv2.VideoCapture(0)
myClassifier=cvzone.Classifier('keras_model/keras_model.h5','labels.txt')

fpsReader=cvzone.FPS()
while True:
    _, img=cap.read()
    predictions = myClassifier.getPrediction(img, scale=1.5)
    #print(predictions, index)
    fps, img =fpsReader.update(img, pos=(450,50))
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    
