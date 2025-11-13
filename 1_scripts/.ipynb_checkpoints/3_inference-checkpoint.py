import os
import re
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
from tqdm import tqdm
import json

# Configuration du device (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paramètres
test_dir = '/marbec-data/FishDeep/identif/detection'  # Dossier contenant les images à analyser
#test_dir = '/marbec-data/FishDeep/training_dataset/test'
batch_size = 32
csv_output_path = '/marbec-data/FishDeep/identif_v2/3_results/3_identif_detection_results.csv'
#csv_output_path = '~/marbec-data/FishDeep/identif/inference_threshold_results_V3.csv'
model_save_path = '/marbec-data/FishDeep/identif_v2/3_results/vit_model.pth'

# Charger les classes depuis le fichier JSON
with open("/marbec-data/FishDeep/identif_v2/3_results/1_classes.json", "r") as f:
    class_to_idx = json.load(f)
# Inverser le dictionnaire 
idx_to_class = {v: k for k, v in class_to_idx.items()}


# Transformations des images (doivent être identiques à celles utilisées lors de l'entraînement)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Taille attendue par ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Chargement du modèle pré-entraîné
num_classes = 158
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()  # Mode évaluation


def filter_images(image_paths):
    fichiers_valides = []
    for fichier in image_paths:
        match = re.search(r'_(\d+\.?\d*)_(\d+\.?\d*)\.jpg$', fichier)
        if match:
            h = float(match.group(1))
            w = float(match.group(2))          
            if w * 1920 * h * 1080 > 1250:
                fichiers_valides.append(fichier)
    return fichiers_valides

# Fonction pour faire de l'inférence sur un batch d'images
def predict_images_from_folder(model, image_folder, batch_size):
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    image_paths = filter_images(image_paths)
    
    #image_paths = [os.path.join(root, file) for root, _, files in os.walk(image_folder) for file in files if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
    all_image_names = []
    all_preds = []
    all_confidences = []

    # Traitement par batch
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
        batch_paths = image_paths[i:i+batch_size]  # Sélection des images du batch
        batch_images = []

        for img_path in batch_paths:
            image = Image.open(img_path).convert("RGB")  # Ouvrir et convertir en RGB
            image = transform(image)  # Appliquer les transformations
            batch_images.append(image)

        if not batch_images:
            continue  # Éviter un batch vide à la fin

        batch_tensor = torch.stack(batch_images).to(device)  # Conversion en tenseur
        with torch.no_grad():
            outputs = model(batch_tensor).logits
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

        # Stocker les résultats
        all_image_names.extend([os.path.basename(p) for p in batch_paths])
        all_preds.extend(preds.cpu().numpy())
        all_confidences.extend([probs[j, preds[j]].item() for j in range(len(preds))])  # Score de confiance max

    return all_image_names, all_preds, all_confidences

# Lancer l'inférence
image_names, pred_labels, confidence_scores = predict_images_from_folder(model, test_dir, batch_size)

# Sauvegarde des résultats dans un CSV
converted_labels = [idx_to_class[idx] for idx in pred_labels]
df_results = pd.DataFrame({
    'Image': image_names,
    'Pred_Label': converted_labels,  # On utilise bien la version convertie ici
    'Confidence_Score': confidence_scores
})

df_results.to_csv(csv_output_path, index=False)
print(f"Test results saved to {csv_output_path}")