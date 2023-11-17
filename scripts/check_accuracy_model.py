from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Fonction pour charger et prétraiter les images
def preprocess_images(images, image_size=48):
    ''' Convertir les images en un format utilisable pour le CNN '''
    num_pixels = image_size * image_size
    images = images / 255.0  # Normalisation
    images = images.reshape(-1, image_size, image_size, 1)  # Redimensionnement
    return images

# Fonction pour lire et prétraiter les données
def load_and_preprocess_data(file_path, is_train=True):
    data = pd.read_csv(file_path)

    if is_train:
        # Extraction et conversion des données en tableau numpy
        X = np.array([np.fromstring(image, sep=' ') for image in data.iloc[:, 1]])
        y = pd.get_dummies(data.iloc[:, 0]).values
    else:
        X = np.array([np.fromstring(image, sep=' ') for image in data.iloc[:, 0]])

    # Prétraitement des images
    X = preprocess_images(X)
    return (X, y) if is_train else X


# Chemin vers le modèle sauvegardé
model_path = 'emotion_model.h5'

# Chemin vers le fichier de données (ajuster selon vos besoins)
data_file_path = 'data/train.csv'  # Utilisez le même fichier que pour l'entraînement

# Charger et prétraiter les données
X, y = load_and_preprocess_data(data_file_path)

# Division en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Charger le modèle
model = load_model(model_path)

# Évaluer le modèle sur l'ensemble de validation
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")
