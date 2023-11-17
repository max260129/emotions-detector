import os
import tensorflow as tf

# Désactiver les avertissements de niveau inférieur de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = FATAL
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import HeNormal
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def preprocess_images(images, image_size=48):
    ''' Convertir les images en un format utilisable pour le CNN '''
    num_pixels = image_size * image_size
    images = images / 255.0  # Normalisation
    images = images.reshape(-1, image_size, image_size, 1)  # Redimensionnement
    return images

# Fonction pour charger et prétraiter les données
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

# Chemins vers les fichiers de données
train_file_path = 'data/train.csv'
test_file_path = 'data/test.csv'

# Charger et prétraiter les données
X_train, y_train = load_and_preprocess_data(train_file_path)
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

# Création du modèle CNN
model = Sequential([
    # Première couche convolutive avec normalisation par lots
    Conv2D(32, (3, 3), kernel_initializer=HeNormal(), input_shape=(48, 48, 1)),
    LeakyReLU(alpha=0.01),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Deuxième couche convolutive
    Conv2D(64, (3, 3), kernel_initializer=HeNormal()),
    LeakyReLU(alpha=0.01),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Troisième couche convolutive
    Conv2D(128, (3, 3), kernel_initializer=HeNormal()),
    LeakyReLU(alpha=0.01),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Aplatissage des caractéristiques pour le fully connected
    Flatten(),
    
    # Dense layer avec plus de neurones
    Dense(128, kernel_initializer=HeNormal()),
    LeakyReLU(alpha=0.01),
    BatchNormalization(),
    Dropout(0.5),
    
    # Couche de sortie avec softmax pour classification multiclasse
    Dense(7, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback pour l'arrêt précoce
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Entraînement du modèle
model.fit(datagen.flow(X_train, y_train, batch_size=64), 
          epochs=15, 
          validation_data=(X_val, y_val), 
          callbacks=[early_stopping])

# Sauvegarde du modèle
model.save('emotion_model.h5')

print("Modèle entraîné et sauvegardé sous 'emotion_model.h5'")
