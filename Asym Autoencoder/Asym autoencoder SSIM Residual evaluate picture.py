# ===============================
#  Autoencoder - Test complet
# ===============================

import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

# ===============================
#  Configuration
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ===============================
#  Modèle Asymétrique
# ===============================
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


# ===============================
#  Chargement des modèles
# ===============================
encoder = Encoder().to(device)
decoder = Decoder().to(device)

encoder.load_state_dict(torch.load(
    "C:/Users/PROMOTECH/OneDrive/Documents/DB/Asym Autoencoder/encoder_asym_works.pth",
    map_location=device
))
decoder.load_state_dict(torch.load(
    "C:/Users/PROMOTECH/OneDrive/Documents/DB/Asym Autoencoder/decoder_asym_works.pth",
    map_location=device
))

encoder.eval()
decoder.eval()
print("✅ Modèles chargés avec succès")

# ===============================
#  Préparation des transforms
# ===============================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def denormalize(t):
    return (t * 0.5) + 0.5

# ===============================
#  Test d'inférence
# ===============================
image_path = "C:/Users/PROMOTECH/OneDrive/Documents/DB/DATABASE/test/22.jpg"
img = Image.open(image_path).convert("RGB")

img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    latent = encoder(img_tensor)                 # vecteur latent
    recon = decoder(latent)                      # reconstruction
    recon_img = denormalize(recon.squeeze()).cpu().permute(1, 2, 0).numpy()

# ===============================
#  Affichage des résultats
# ===============================
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Reconstruit")
plt.imshow(recon_img)
plt.axis("off")

plt.tight_layout()
plt.show()

# ===============================
#  Sauvegarde du vecteur latent
# ===============================
Path("latents").mkdir(exist_ok=True)
latent_np = latent.squeeze().cpu().numpy()

with open("latents/latent_from_loaded_model.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(latent_np.flatten())

print("✅ Latent vector sauvegardé et test terminé")
