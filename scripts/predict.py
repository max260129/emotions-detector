import os
import tensorflow as tf

# Désactiver les avertissements de niveau inférieur de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = FATAL
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
    
    # Séparer les pixels par des espaces et les convertir en entiers
    pixels = data['pixels'].apply(lambda x: [int(pixel) for pixel in x.split()])
    
    # Convertir en un tableau NumPy
    images = np.vstack(pixels.values)
    
    # Vérifier que chaque image contient 2304 pixels
    if images.shape[1] != 2304:
        raise ValueError(f"Chaque image doit contenir 2304 pixels, mais trouvé {images.shape[1]} pixels.")

    # Redimensionner les images pour qu'elles correspondent à la forme attendue par le modèle
    images = preprocess_images(images)
    
    if is_train:
        # Obtenir les étiquettes si c'est un ensemble d'entraînement
        y = pd.get_dummies(data['emotion']).values
        return images, y
    else:
        # Sinon, retourner seulement les images
        return images

# Chemin vers le modèle sauvegardé
model_path = 'emotion_model.h5'

# Chemin vers le fichier de données (ajuster selon vos besoins)
data_file_path = 'filegive/test_with_emotions.csv'  # Utilisez le même fichier que pour l'entraînement

# Charger et prétraiter les données
X, y = load_and_preprocess_data(data_file_path)

# Division en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Charger le modèle
model = load_model(model_path)

# Évaluer le modèle sur l'ensemble de validation
loss, accuracy = model.evaluate(X_val, y_val)

final_num = round(accuracy * 100)
print(f"Accuracy on test set: {final_num}%")
