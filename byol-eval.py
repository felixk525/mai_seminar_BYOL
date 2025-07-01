import os
import torch
import random
import numpy as np
from torchvision import models, transforms
from torchvision.datasets import EuroSAT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# Dataset Setup
class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.filepaths = [os.path.join(root, fname) for fname in os.listdir(root)
                          if fname.lower().endswith(('.tif', '.tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

# Transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

# Load  seperate labeled Dataset for evaluation
dataset = EuroSAT(root='.', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, persistent_workers= True, num_workers=4)

# Load Backbone (Trained and Random)
def get_backbone(trained=True, path = './improved-satellite-resnet.pt'):
    model = models.resnet50(weights=None)
    if trained:
        model.load_state_dict(torch.load(path, weights_only=True)) #, map_location='cpu'
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final classifier
    model.eval()
    return model

# Feature Extraction - for later use with Linear Classifier
def extract_features(backbone, dataloader, device):
    backbone.eval()
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            out = backbone(imgs).squeeze()
            features.append(out.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    return np.vstack(features), np.array(labels)

# Evaluation Function
def evaluate(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

# MAIN: Compare trained vs untrained
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_encoder = get_backbone(trained=True, path='./best-satellite-resnet.pt').to(device)
    untrained_encoder = get_backbone(trained=False).to(device)

    X_trained, y1 = extract_features(trained_encoder, dataloader, device)
    X_untrained, y2 = extract_features(untrained_encoder, dataloader, device)

    print(f"This will take a while...")
    # Train logistic regression
    X_train, X_test, y_train, y_test = train_test_split(X_trained, y1, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000).fit(X_train, y_train) # if no convergion try 5000 iter for convergion
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy with BYOL-trained encoder: {acc:.4f}")

    # Train logistic regression (untrained features)
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_untrained, y2, test_size=0.2, random_state=42)
    scaler_u = StandardScaler()
    X_train_u = scaler_u.fit_transform(X_train_u)
    X_test_u = scaler_u.transform(X_test_u)
    clf_u = LogisticRegression(max_iter=5000).fit(X_train_u, y_train_u) # if no convergion try 5000 iter for convergion
    acc_u = accuracy_score(y_test_u, clf_u.predict(X_test_u))
    print(f"Accuracy with untrained encoder: {acc_u:.4f}")
