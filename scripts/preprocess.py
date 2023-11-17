import os
import tensorflow as tf

# Désactiver les avertissements de niveau inférieur de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = FATAL
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Chemins vers les fichiers de données
train_file_path = 'data/train.csv'
test_file_path = 'data/test.csv'

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

# Charger et prétraiter les données d'entraînement
X_train, y_train = load_and_preprocess_data(train_file_path)

# Charger et prétraiter les données de test
X_test = load_and_preprocess_data(test_file_path, is_train=False)

# Division en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Génération de données supplémentaires pour l'entraînement
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

print("Prétraitement terminé. Données prêtes pour l'entraînement et l'évaluation.")

# Fonction pour afficher des images avec leurs étiquettes
def display_sample_images(images, labels, num_images=5):
    ''' Affiche un échantillon d'images avec leurs étiquettes '''
    fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
    axes = axes.flatten()

    for img, lbl, ax in zip(images[:num_images], labels[:num_images], axes):
        ax.imshow(img.squeeze(), cmap='gray')  # Les images sont en niveaux de gris
        ax.axis('off')
        ax.set_title(lbl)

    plt.tight_layout()
    plt.show()


"""
# Exemple d'affichage des images d'entraînement
# Convertir les étiquettes one-hot en étiquettes de classe
y_train_labels = [np.argmax(label) for label in y_train]
display_sample_images(X_train, y_train_labels)
"""