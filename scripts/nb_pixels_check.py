import pandas as pd

# Chemin vers le fichier CSV
file_path = 'filegive/test_with_emotions.csv'

# Charger les données
data = pd.read_csv(file_path)

# Supposons que les images sont dans la deuxième colonne
pixel_counts = data.iloc[:, 2].apply(lambda x: len(x.split()))

# Vérifier si toutes les images ont le même nombre de pixels
if pixel_counts.nunique() == 1:
    print(f"Toutes les images ont {pixel_counts.iloc[0]} pixels.")
else:
    print("Inconsistance dans le nombre de pixels des images.")
