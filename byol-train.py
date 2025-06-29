import os
import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from byol_pytorch import BYOL
from PIL import Image
from tqdm import tqdm

# Subset dataset path
dataset_path = r"C:\PCodetmp\mai_seminar_byol\rgb_subset"

# Formatting
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),  # scales to [0, 1]
])

class FlatFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.filepaths = [os.path.join(root, fname) for fname in os.listdir(root)
                          if fname.lower().endswith(('.tif', '.tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    
if __name__ == "__main__": # Avoid worker errors
    dataset = FlatFolderDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, persistent_workers= True, num_workers=4) # multiprocessing

    # BYOL leaner
    resnet = models.resnet50(weights=None)  # use True for pretrained weights - set to None since goal is to train personal

    learner = BYOL(
        resnet,
        image_size=256,
        hidden_layer='avgpool',
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    # training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learner = learner.to(device)
    resnet.to(device)
    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler(device='cuda') # mixed precision

    best_loss = float('inf')

    for epoch in range(10):
        epoch_loss = 0
        for images in tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False):
            images = images.to(device)
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Save best model (in case of performance degradation)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(resnet.state_dict(), './best-satellite-resnet.pt')
            print(f"Saved new best model at epoch {epoch + 1} with loss {best_loss:.4f}")

    # Optional final save (e.g., for last epoch model)
    torch.save(resnet.state_dict(), './improved-satellite-resnet.pt')
    print("Final model saved.")

