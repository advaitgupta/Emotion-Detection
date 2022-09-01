from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(
    '/Users/advaitgupta/Desktop/Projects & Study/Projects/Personal/Emotion Detection Model/Pre Trained Model/haarcascade_frontalface_default.xml')
classifier = load_model(
    '/Users/advaitgupta/Desktop/Projects & Study/Projects/Personal/Emotion Detection Model/Pre Trained Model/Emotion_Model1.h5')

class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

capture = cv2.VideoCapture(0)

while True:
    r, img = capture.read()
    labels = []
    faces = face_classifier.detectMultiScale(img, 1.3, 7)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img1 = img[y:y + h, x:x + w]
        img1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_AREA)

        if np.sum([img1]) != 0:
            roi = img1.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            predictions = classifier.predict(roi)[0]
            label = class_labels[predictions.argmax()]
            label_position = (x, y)
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(img, 'No Face Present', (20, 20), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Emotion Detection', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
capture.release()
cv2.destroyAllWindows()
