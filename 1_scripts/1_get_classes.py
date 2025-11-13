import json
from torchvision import datasets

train_dir = '/marbec-data/FishDeep/identif_v2/2_training_dataset/train' 
train_dataset = datasets.ImageFolder(root=train_dir)  
class_names = train_dataset.classes  # Liste des noms de classes

# Sauvegarde des classes
with open("/marbec-data/FishDeep/identif_v2/3_results/1_classes.json", "w") as f:
    json.dump(train_dataset.class_to_idx, f)

print("Classes sauvegardées :", train_dataset.class_to_idx)