import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #i am using CPU 
model_drop_out = load_model('model_data_aug.h5')
labels = ['cat', 'dog']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=2)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = frame[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (150, 150))
        input_data = np.expand_dims(resized_face, axis=0) / 255.0
        prediction = model_drop_out.predict(input_data)
        predicted_label = labels[int(round(prediction[0][0]))]
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Cat/Dog Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

