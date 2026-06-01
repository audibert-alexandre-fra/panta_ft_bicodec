import os
import shutil
import random

# Dossiers source et destination
source_dir = "filter_data"
dest_dir = "test_data"

# Nombre de fichiers à déplacer
num_files_to_move = 100

# Création du dossier destination s'il n'existe pas
os.makedirs(dest_dir, exist_ok=True)

# Liste des fichiers dans le dossier source
files = [
    f for f in os.listdir(source_dir)
    if os.path.isfile(os.path.join(source_dir, f))
]

# Vérifie qu'il y a assez de fichiers
if len(files) < num_files_to_move:
    raise ValueError(
        f"Seulement {len(files)} fichiers disponibles dans {source_dir}"
    )

# Sélection aléatoire de 100 fichiers
selected_files = random.sample(files, num_files_to_move)

# Déplacement des fichiers
for file_name in selected_files:
    src_path = os.path.join(source_dir, file_name)
    dest_path = os.path.join(dest_dir, file_name)

    shutil.move(src_path, dest_path)

print(f"{num_files_to_move} fichiers déplacés de '{source_dir}' vers '{dest_dir}'.")