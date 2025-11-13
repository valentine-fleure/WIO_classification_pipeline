import os
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from collections import defaultdict
import random

# Configuration du device (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paramètres
train_dir = '~/marbec-data/FishDeep/identif_v2/2_training_dataset/train' # Dossier des images d'entraînement
test_dir = '~/marbec-data/FishDeep/identif_v2/2_training_dataset/test'    # Dossier des images de test
batch_size = 32
learning_rate = 1e-5
epochs = 5
csv_output_path = '/marbec-data/FishDeep/identif_v2/3_results/2_inference_test.csv'
model_save_path = '/marbec-data/FishDeep/identif_v2/3_results/vit_model.pth'

# Transformations des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Taille attendue par ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Chargement du dataset
full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Séparation 80/20 par classe
train_data_by_class = defaultdict(list)

# Organiser les indices des images par classe
for idx, (image, label) in enumerate(full_train_dataset.imgs):
    train_data_by_class[label].append(idx)

# Séparer les indices par classe
train_indices = []
val_indices = []

for label, indices in train_data_by_class.items():
    random.shuffle(indices)  # Mélanger les indices de chaque classe
    split_idx = int(0.8 * len(indices))  # 80% pour l'entraînement
    train_indices.extend(indices[:split_idx])  # 80% des indices pour l'entraînement
    val_indices.extend(indices[split_idx:])  # 20% des indices pour la validation

# Créer les sous-ensembles d'entraînement et de validation
train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
val_subset = torch.utils.data.Subset(full_train_dataset, val_indices)

# Chargement des DataLoaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Nombre de classes
num_classes = len(full_train_dataset.classes)

# Chargement du modèle ViT pré-entraîné et préparation
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)
model.to(device)

# Optimiseur et fonction de perte
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Fonction pour évaluer le modèle sur un DataLoader (train, val ou test)
def evaluate_train_val(model, dataloader, dataset):
    model.eval()
    y_true = []
    y_pred = []
    confidence_scores = []
    image_names = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            _, preds = torch.max(probs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
            # Enregistrer les scores de confiance et les noms d'images
            for idx in range(len(images)):
                image_name = dataset.imgs[dataloader.dataset.indices[idx]][0].split('/')[-1]
                image_names.append(image_name)
                confidence_scores.append(probs[idx].cpu().numpy())

    return y_true, y_pred, confidence_scores, image_names

def evaluate_test(model, dataloader, dataset):
    model.eval()
    y_true = []
    y_pred = []
    confidence_scores = []
    image_names = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader): 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            probs = torch.nn.functional.softmax(outputs, dim=1)

            _, preds = torch.max(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Index réel des images dans le dataset
            batch_start_idx = batch_idx * dataloader.batch_size
            for i in range(len(images)):
                real_idx = batch_start_idx + i  # Calcul de l'index réel
                if real_idx < len(dataset.imgs):  # Vérification pour éviter un dépassement
                    image_name = dataset.imgs[real_idx][0].split('/')[-1]  # Récupération du nom correct
                    image_names.append(image_name)
                    confidence_scores.append(probs[i].cpu().numpy())

    return y_true, y_pred, confidence_scores, image_names


# Entraînement du modèle
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Évaluation sur la validation
    y_true_val, y_pred_val, confidence_scores_val, image_names_val = evaluate_train_val(model, val_loader, full_train_dataset)
    acc_val = accuracy_score(y_true_val, y_pred_val)
    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader)} - Validation Accuracy: {acc_val}")

# Sauvegarde des poids du modèle
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")

model.load_state_dict(torch.load(model_save_path, weights_only=True))

###### Evaluation ######
# Évaluation finale sur les données de test
y_true_test, y_pred_test, confidence_scores_test, image_names_test = evaluate_test(model, test_loader, test_dataset)
acc_test = accuracy_score(y_true_test, y_pred_test)
report_test = classification_report(y_true_test, y_pred_test, target_names=test_dataset.classes)
report_test = classification_report(y_true_test, y_pred_test)

print(f"Test Accuracy: {acc_test}")
print("Test Classification Report:")
print(report_test)

# Sauvegarde des résultats dans un fichier CSV
df_test = pd.DataFrame({
    'Image': image_names_test,
    'True_Label': [test_dataset.classes[label] for label in y_true_test],
    'Pred_Label': [test_dataset.classes[label] for label in y_pred_test],
    'Confidence_Score': [max(score) for score in confidence_scores_test]  # Score de confiance le plus élevé
})

df_test.to_csv(csv_output_path, index=False)
print(f"Test results saved to {csv_output_path}")
