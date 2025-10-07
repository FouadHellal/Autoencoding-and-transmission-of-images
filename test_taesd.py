#TO USE THIS SCRIPT YOU HAVE TO CLONE THE ORIGINAL REPO FIRST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from taesd import TAESD  # from madebyollin/taesd repo
from pytorch_msssim import ssim

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Load pretrained TAESD
taesd = TAESD().to(device)
taesd.eval()

# 3. Load and preprocess your image (resize to 256x256)
img_path = "C:/Users/PROMOTECH/OneDrive/Documents/DB/DATABASE/test/22.jpg"
img = Image.open(img_path).convert("RGB")
img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

# 4. Encode -> latent
with torch.no_grad():
    latent = taesd.encoder(img_tensor)
print("Latent shape:", latent.shape)

# 5. Decode -> reconstruction
with torch.no_grad():
    recon = taesd.decoder(latent).clamp(0, 1)

# 6. Measure quality (optional)
quality = ssim(img_tensor, recon, data_range=1.0)
print("SSIM:", quality.item())

# 7. Visualize
img_np = img_tensor.squeeze().permute(1,2,0).cpu().numpy()
recon_np = recon.squeeze().permute(1,2,0).cpu().numpy()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Reconstructed (TAESD)")
plt.imshow(recon_np)
plt.axis("off")

plt.tight_layout()
plt.show()

# 8. Save latent to CSV (optional)
import csv, os
os.makedirs("latents", exist_ok=True)
latent_np = latent.squeeze().cpu().numpy()
with open("latents/latent_taesd.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(latent_np.flatten())

print("Latent vector saved at latents/latent_taesd.csv")

"""-----------------------for 128*128 images 1024 latent--------------------------------------------------------------"""
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from taesd import TAESD
from pytorch_msssim import ssim
import csv, os

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Charger le modèle TAESD
taesd = TAESD().to(device)
taesd.eval()

# 3. Charger et redimensionner ton image en 128x128
img_path = "C:/Users/PROMOTECH/OneDrive/Documents/DB/DATABASE/test/22.jpg"
img = Image.open(img_path).convert("RGB")
img = img.resize((128, 128))  # <--- redimensionnement ici
img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

# 4. Encoder -> latent
with torch.no_grad():
    latent = taesd.encoder(img_tensor)
print("Latent shape:", latent.shape)  # devrait être [1, 4, 32, 32]

# 5. Décoder -> reconstruction
with torch.no_grad():
    recon = taesd.decoder(latent).clamp(0, 1)

# 6. Qualité (optionnelle)
quality = ssim(img_tensor, recon, data_range=1.0)
print("SSIM:", quality.item())

# 7. Visualisation
img_np = img_tensor.squeeze().permute(1,2,0).cpu().numpy()
recon_np = recon.squeeze().permute(1,2,0).cpu().numpy()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original 128×128")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Reconstruit (TAESD)")
plt.imshow(recon_np)
plt.axis("off")

plt.tight_layout()
plt.show()

# 8. Sauvegarde du latent
os.makedirs("latents", exist_ok=True)
latent_np = latent.squeeze().cpu().numpy()
with open("latents/latent_taesd_128.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(latent_np.flatten())

print("✅ Latent vector saved at latents/latent_taesd_128.csv")

