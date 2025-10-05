import os
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_msssim import ssim
from tqdm import tqdm

# ---- Configuration de base ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True

# ---- Dataset personnalisé ----
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

# ---- Transforms ([-1, 1] avec Tanh) ----
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def denormalize(t):
    return (t * 0.5) + 0.5

train_dataset = CustomImageDataset("C:/Users/PROMOTECH/OneDrive/Documents/DB/DATABASE/train", transform)
test_dataset = CustomImageDataset("C:/Users/PROMOTECH/OneDrive/Documents/DB/DATABASE/test", transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---- Modèle Asymétrique CAE ----
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 96, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(96, 64, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 48, 1),
            nn.LeakyReLU(),
            nn.Conv2d(48, 8, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 48, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(48, 256, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 48, 3, 2, 1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(48, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.decoder(z)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder().to(device)

# ---- Fonction de perte combinée ----
def combined_loss(output, target):
    residual = nn.functional.l1_loss(output, target)
    ssim_loss = 1 - ssim(output, target, data_range=1.0, size_average=True)
    return residual + 0.5 * ssim_loss

optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 80
loss_history = []

# ---- Entraînement ----
for epoch in range(epochs):
    model.train()
    total_loss = 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for batch in pbar:
            imgs, _ = batch
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = combined_loss(recon, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    avg = total_loss / len(train_loader)
    loss_history.append(avg)
    print(f"Epoch {epoch+1} finished with loss: {avg:.6f}")

# ---- Affichage du loss ----
plt.figure()
plt.plot(loss_history, label="Combined Loss")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig("C:/Users/PROMOTECH/OneDrive/Documents/DB/models/loss_plot.png")
plt.show()

# ---- Visualisation des reconstructions ----
model.eval()
with torch.no_grad():
    sample_batch, _ = next(iter(test_loader))
    sample_batch = sample_batch.to(device)
    reconstructed = model(sample_batch)

    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for i in range(6):
        axes[0, i].imshow(denormalize(sample_batch[i]).cpu().permute(1, 2, 0))
        axes[0, i].axis('off')
        axes[1, i].imshow(denormalize(reconstructed[i]).cpu().permute(1, 2, 0))
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig("C:/Users/PROMOTECH/OneDrive/Documents/DB/models/reconstruction_samples.png")
    plt.show()

# ---- Sauvegarde ----
os.makedirs("C:/Users/PROMOTECH/OneDrive/Documents/DB/models", exist_ok=True)
torch.save(model.encoder.state_dict(), "C:/Users/PROMOTECH/OneDrive/Documents/DB/models/encoder_asym_works.pth")
torch.save(model.decoder.state_dict(), "C:/Users/PROMOTECH/OneDrive/Documents/DB/models/decoder_asym_works.pth")
print("Encoder et Decoder sauvegardés avec succès.")

# ---- Test d'inférence sur une image et export du vecteur latent ----
image_path = "C:/Users/PROMOTECH/OneDrive/Documents/DB/DATABASE/test/tst.jpg"
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    latent = model.encoder(img_tensor).squeeze().cpu().numpy()
    recon_img = model.decoder(torch.tensor(latent).unsqueeze(0).to(device)).squeeze().cpu()
    recon_img = denormalize(recon_img).permute(1, 2, 0).numpy()

# Sauvegarde du vecteur latent en CSV
Path("latents").mkdir(exist_ok=True)
with open("latents/latent_vector.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(latent.flatten())

# Affichage de la reconstruction
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Reconstruit")
plt.imshow(recon_img)
plt.axis('off')
plt.tight_layout()
plt.show()
print("Latent vector saved and reconstruction done.")