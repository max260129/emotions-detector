import cv2
from keras.models import load_model
import numpy as np
import datetime

# Charger le modèle
model = load_model('emotion_model.h5')

# Liste des émotions (à adapter selon votre modèle)
emotions = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust', 'Neutral']

# Fonction pour prétraiter les frames
def preprocess_frame(frame, image_size=48):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (image_size, image_size))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, image_size, image_size, 1))
    return reshaped

# Fonction pour prédire l'émotion sur une frame
def predict_emotion(frame):
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    emotion_probability = np.max(predictions)
    emotion_index = np.argmax(predictions)
    emotion_name = emotions[emotion_index]
    return emotion_name, round(emotion_probability * 100, 2)

# Lire la vidéo
video_path = 'vid.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Horodatage
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # Afficher le message de prétraitement
    print(f"\nPreprocessing ...")

    # Prédire l'émotion
    emotion_name, emotion_probability = predict_emotion(frame)
    print(f"{timestamp} : {emotion_name} , {emotion_probability}%")

    # Afficher le frame (optionnel)
    cv2.imshow('Video', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
