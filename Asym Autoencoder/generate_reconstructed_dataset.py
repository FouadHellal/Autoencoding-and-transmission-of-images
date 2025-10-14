import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# ===============================
#  Configuration
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

input_root = r"C:\Users\PROMOTECH\OneDrive\Documents\DB\SPLIT"
output_root = r"C:\Users\PROMOTECH\OneDrive\Documents\DB\SPLIT_RECONSTRUCTED"

os.makedirs(output_root, exist_ok=True)

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
    r"C:\Users\PROMOTECH\OneDrive\Documents\DB\Asym Autoencoder\encoder_asym_works.pth",
    map_location=device
))
decoder.load_state_dict(torch.load(
    r"C:\Users\PROMOTECH\OneDrive\Documents\DB\Asym Autoencoder\decoder_asym_works.pth",
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
#  Fonction principale
# ===============================
def process_dataset(split_name):
    input_split = os.path.join(input_root, split_name)
    output_split = os.path.join(output_root, split_name)
    os.makedirs(output_split, exist_ok=True)

    # parcourir chaque classe
    for class_name in os.listdir(input_split):
        class_input = os.path.join(input_split, class_name)
        class_output = os.path.join(output_split, class_name)
        os.makedirs(class_output, exist_ok=True)

        # parcourir chaque image
        image_files = [f for f in os.listdir(class_input) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        print(f"Processing {split_name}/{class_name} ({len(image_files)} images)")

        for img_name in tqdm(image_files, desc=f"{split_name}/{class_name}"):
            img_path = os.path.join(class_input, img_name)
            img = Image.open(img_path).convert("RGB")

            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                latent = encoder(img_tensor)
                recon = decoder(latent)
                recon_img = denormalize(recon.squeeze()).cpu().permute(1, 2, 0).numpy()
                recon_img = np.clip(recon_img, 0, 1)
                recon_img = (recon_img * 255).astype(np.uint8)
                recon_pil = Image.fromarray(recon_img)

            save_path = os.path.join(class_output, img_name)
            recon_pil.save(save_path)

    print(f"✅ Reconstruction terminée pour {split_name}")


# ===============================
#  Exécution complète
# ===============================
for split in ["train", "val", "test"]:
    process_dataset(split)

print("\n✅ Reconstruction complète du dataset terminée !")
print(f"Les images sont enregistrées dans : {output_root}")
