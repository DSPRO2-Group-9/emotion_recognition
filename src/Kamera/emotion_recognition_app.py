#comand to create .exe in terminal: pyinstaller --onefile --windowed --add-data "C:/Emotion_Recognition/src/kamera/haarcascade_frontalface_default.xml;." --add-data "C:/Emotion_Recognition/model/240606_mobilenetv2_augmentation_model.keras;." C:/Emotion_Recognition/src/Kamera/emotion_recognition_app.py

import tkinter as tk
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

import sys
import os

print(os.getcwd())

from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

if getattr(sys, 'frozen', False):
    # Running as a standalone executable
    exe_dir = sys._MEIPASS
    haar_cascade_path = os.path.join(exe_dir, 'haarcascade_frontalface_default.xml')
    model_path = os.path.join(exe_dir, '240620_mobilenetv2_model.keras')
else:
    # Running in development mode
    exe_dir = os.path.dirname(__file__)
    haar_cascade_path = 'C:/Emotion_Recognition/src/kamera/haarcascade_frontalface_default.xml'
    model_path = 'C:/Emotion_Recognition/model/240620_mobilenetv2_model.keras'


# Construct the paths to the bundled files
haar_cascade_path = os.path.join(exe_dir, 'haarcascade_frontalface_default.xml')
model_path = os.path.join(exe_dir, '240620_mobilenetv2_model.keras')


face_classifier = cv2.CascadeClassifier(haar_cascade_path)

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition")
        self.root.geometry("800x600")

        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill="both", expand=True)

        self.header_frame = tk.Frame(self.main_frame, bg="#333")
        self.header_frame.pack(fill="x")

        self.welcome_label = tk.Label(self.header_frame, text="Emotion Recognition", font=("Helvetica", 24, "bold"), bg="#333", fg="white")
        self.welcome_label.pack(pady=20)

        self.instruction_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.instruction_frame.pack(fill="x", pady=20)

        self.instruction_label = tk.Label(self.instruction_frame, text="Welcome to our Emotion Recognition Project! Here you can start the video capture and see your emotions recognized in real-time.", font=("Helvetica", 14), bg="#f0f0f0", wraplength=600)
        self.instruction_label.pack(pady=10)

        self.hint_label = tk.Label(self.instruction_frame, text="Hint: Press 'q' to quit the camera.", font=("Helvetica", 12), bg="#f0f0f0")
        self.hint_label.pack(pady=10)

        self.video_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.video_frame.pack(fill="both", expand=True)

        self.video_label = tk.Label(self.video_frame, bg="#f0f0f0")
        self.video_label.pack(pady=20)

        self.capture_button = tk.Button(self.video_frame, text="Start Capture", command=self.capture_video, font=("Helvetica", 14), bg="#FFC107", fg="white", activebackground="#FFA07A", activeforeground="white", padx=10, pady=5)
        self.capture_button.pack(pady=20)

        self.classifier = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


    def samplewise_standardization(self, X):
        X = X.astype('float32')
        X = X - np.mean(X, axis=(1, 2), keepdims=True)
        X = X / np.std(X, axis=(1, 2), keepdims=True)
        return X

    def capture_video(self):
        cap = cv2.VideoCapture(0)

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
                    roi = self.samplewise_standardization(roi)
                    roi = np.expand_dims(roi,axis=0)
                    roi= np.stack([roi] * 3, axis=3)
                    
                    prediction = self.classifier.predict(roi)[0]
                    top3_indices = np.argsort(prediction)[-3:][::-1]
                    top3_labels = [self.emotion_labels[i] for i in top3_indices]
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

if __name__ == '__main__':
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()