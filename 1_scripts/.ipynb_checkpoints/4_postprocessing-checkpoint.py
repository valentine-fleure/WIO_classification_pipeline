import pandas as pd
import os
import shutil

# Chemins à adapter
csv_path = "/marbec-data/FishDeep/identif_v2/post_pro_folder.csv"
source_image_dir = "/marbec-data/FishDeep/identif/detection"  # répertoire où sont les images originales
destination_base_dir = "/marbec-data/FishDeep/identif_v2/4_post_processing_folders"

# Chargement du CSV
df = pd.read_csv(csv_path)

# Parcours des lignes du fichier
for idx, row in df.iterrows():
    image_name = row['Image']
    pred_label = str(row['Pred_Label'])
    folder = str(row['folder'])

    # Création du chemin de destination
    destination_dir = os.path.join(destination_base_dir, pred_label, folder)
    os.makedirs(destination_dir, exist_ok=True)

    # Chemin source et destination du fichier
    src_path = os.path.join(source_image_dir, image_name)
    dst_path = os.path.join(destination_dir, image_name)

    # Copie si le fichier source existe
    if os.path.isfile(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"Image non trouvée : {src_path}")