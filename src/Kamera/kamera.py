

from tensorflow.keras.models import load_model # type: ignore

from tensorflow.keras.preprocessing.image import img_to_array # type: ignore


import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier("/Emotion_Recognition/src/kamera/haarcascade_frontalface_default.xml")
classifier =load_model("/Emotion_Recognition/model/240606_mobilenetv2_augmentation_model.keras")

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


cap = cv2.VideoCapture(0)


def samplewise_standardization(X):
    return (X - np.mean(X)) / np.std(X)


while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(96,96),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float32')
            roi = img_to_array(roi)
            roi = samplewise_standardization(roi)
            roi = np.expand_dims(roi,axis=0)
            roi= np.stack([roi] * 3, axis=3)

            prediction = classifier.predict(roi)[0]
            top3_indices = np.argsort(prediction)[-3:][::-1]
            top3_labels = [emotion_labels[i] for i in top3_indices]
            top3_probabilities = [prediction[i] for i in top3_indices]

            for i in range(3):
                label = top3_labels[i]
                probability = top3_probabilities[i]
                label_position = (x, y - 30 * (3 - i))

                if i == 0:
                    text_color = (0, 255, 0)

                else:
                    text_color = (0, 255, 255)

                cv2.putText(frame, f"{label}: {probability*100:.2f}%", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)


        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()