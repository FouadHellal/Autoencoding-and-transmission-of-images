import os
import csv
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pytorch_msssim import ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchinfo import summary

# ----------------------------- Configuration ---------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    try:
        print(torch.cuda.get_device_name(0))
    except Exception:
        pass

input_dim = 128
batch_size = 16
learning_rate = 1e-4
num_epochs = 500
train_ratio, val_ratio = 0.6, 0.2
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

data_dir = r"C:/Users/PROMOTECH/OneDrive/Documents/DB/Potato Leaf Disease Dataset in Uncontrolled Environment"
models_dir = "models1024"
os.makedirs(models_dir, exist_ok=True)

# ----------------------------- Dataset ---------------------------------------
class RGB_Dataset(Dataset):
    def __init__(self, image_dir, input_dim, extensions=("png", "jpg", "jpeg")):
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for fname in files:
                if fname.lower().endswith(extensions):
                    self.image_paths.append(os.path.join(root, fname))

        self.transform = transforms.Compose([
            transforms.Resize((input_dim, input_dim)),
            transforms.ToTensor(),
            # Normalize to [-1, 1] to match tanh decoder output
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# ----------------------------- Model -----------------------------------------

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 32, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 4, 1)  # was 2 → now 4 channels → (B,4,16,16) → 1024 latent
        )

    def forward(self, x):
        x = self.encoder(x)  # (B, 4, 16, 16)
        return torch.flatten(x, 1)  # (B, 1024)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 32, 1), nn.LeakyReLU(inplace=True),  # input channels updated (was 2)
            nn.ConvTranspose2d(32, 256, 1), nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.ConvTranspose2d(128, 96, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.ConvTranspose2d(96, 48, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.ConvTranspose2d(48, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, 4, 16, 16)  # was 2 → now 4
        return self.decoder(z)

# ----------------------------- Losses & Utilities -----------------------------
def sobel_gradients(img):
    """
    Compute Sobel gradients on luminance; img in [-1,1], shape (B,3,H,W).
    Returns gradient magnitude map (B,1,H,W).
    """
    # convert to luminance (simple average; robust/speed)
    gray = img.mean(dim=1, keepdim=True)

    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1,-2,-1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-12)
    return mag

def kl_sparsity(latent, rho=0.05, eps=1e-8):
    """
    KL divergence between desired sparsity rho and average activation of sigmoid(latent).
    latent: (B, D)
    """
    a = torch.sigmoid(latent)  # (0,1)
    rho_hat = a.mean(dim=0)    # per-unit mean over batch
    term1 = rho * torch.log((rho + eps) / (rho_hat + eps))
    term2 = (1 - rho) * torch.log((1 - rho + eps) / (1 - rho_hat + eps))
    kl = term1 + term2
    return kl.mean()  # average over units

def calculate_psnr(original, reconstructed, data_range=2.0):
    """
    PSNR for tensors in [-1,1] -> data_range=2.0
    """
    mse = F.mse_loss(original, reconstructed)
    if mse.item() == 0:
        return 100.0
    return 20 * math.log10(data_range) - 10 * math.log10(mse.item())

@torch.no_grad()
def evaluate_metrics(loader, encoder, decoder, device):
    encoder.eval(); decoder.eval()
    total_mse, total_ssim = 0.0, 0.0
    n = 0
    for imgs in loader:
        imgs = imgs.to(device)
        z = encoder(imgs)
        rec = decoder(z)
        mse = F.mse_loss(imgs, rec, reduction='mean')
        ssim_val = ssim(imgs, rec, data_range=2.0, size_average=True)
        total_mse += mse.item() * imgs.size(0)
        total_ssim += ssim_val.item() * imgs.size(0)
        n += imgs.size(0)
    avg_mse = total_mse / max(1, n)
    avg_ssim = total_ssim / max(1, n)
    psnr = 20 * math.log10(2.0) - 10 * math.log10(max(avg_mse, 1e-12))
    return psnr, avg_ssim

# ----------------------------- Loss Weights & Sparsity Settings ---------------
# You can tune these to emphasize different properties.
w_mse = 1.0
w_mae = 0.25
w_ssim = 0.5
w_edge = 0.1
w_l1_sparsity = 1e-3
w_kl_sparsity = 1e-3
rho_target = 0.05

# ----------------------------- Data Loading -----------------------------------
dataset = RGB_Dataset(data_dir, input_dim)
train_size = int(len(dataset) * train_ratio)
val_size = int(len(dataset) * val_ratio)
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size],
                                            generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ----------------------------- Initialize -------------------------------------
encoder = Encoder().to(device)
decoder = Decoder().to(device)
optim_params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(optim_params, lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Useful info
print(summary(encoder, input_size=(1, 3, input_dim, input_dim)))
print(summary(decoder, input_size=(1, 1024)))

latent_dim = 1024
num_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
print(f"Latent dimension: {latent_dim}, Total parameters: {num_params:,}")

# ----------------------------- Loss Tracking ----------------------------------
loss_names = ["total", "mse", "mae", "ssim", "edge", "l1_sparse", "kl_sparse"]
train_hist = {k: [] for k in loss_names}
val_hist   = {k: [] for k in loss_names}

best_val = float('inf')
best_epoch = -1

# ----------------------------- Training Loop ----------------------------------
for epoch in range(1, num_epochs + 1):
    encoder.train()
    decoder.train()

    # Running sums
    sums = {k: 0.0 for k in loss_names}
    count = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{num_epochs}")
    for imgs in pbar:
        imgs = imgs.to(device)
        optimizer.zero_grad()

        z = encoder(imgs)
        rec = decoder(z)

        # Components
        mse_loss = F.mse_loss(rec, imgs)
        mae_loss = F.l1_loss(rec, imgs)
        ssim_loss = 1.0 - ssim(imgs, rec, data_range=2.0, size_average=True)
        edge_loss = F.l1_loss(sobel_gradients(rec), sobel_gradients(imgs))
        l1_sparsity = torch.mean(torch.abs(z))
        kl_sparse = kl_sparsity(z, rho=rho_target)

        total_loss = (
            w_mse * mse_loss +
            w_mae * mae_loss +
            w_ssim * ssim_loss +
            w_edge * edge_loss +
            w_l1_sparsity * l1_sparsity +
            w_kl_sparsity * kl_sparse
        )

        total_loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        sums["total"] += total_loss.item() * bs
        sums["mse"]   += mse_loss.item()     * bs
        sums["mae"]   += mae_loss.item()     * bs
        sums["ssim"]  += ssim_loss.item()    * bs
        sums["edge"]  += edge_loss.item()    * bs
        sums["l1_sparse"] += l1_sparsity.item() * bs
        sums["kl_sparse"] += kl_sparse.item()   * bs
        count += bs

        pbar.set_postfix({
            "tot": f"{(sums['total']/count):.4f}",
            "mse": f"{(sums['mse']/count):.4f}",
            "ssim": f"{(sums['ssim']/count):.4f}"
        })

    # Averages
    for k in loss_names:
        train_hist[k].append(sums[k] / max(1, count))

    # Validation
    encoder.eval(); decoder.eval()
    sums_val = {k: 0.0 for k in loss_names}
    count_val = 0
    with torch.no_grad():
        for imgs in val_loader:
            imgs = imgs.to(device)
            z = encoder(imgs)
            rec = decoder(z)

            mse_loss = F.mse_loss(rec, imgs)
            mae_loss = F.l1_loss(rec, imgs)
            ssim_loss = 1.0 - ssim(imgs, rec, data_range=2.0, size_average=True)
            edge_loss = F.l1_loss(sobel_gradients(rec), sobel_gradients(imgs))
            l1_sparsity = torch.mean(torch.abs(z))
            kl_sparse = kl_sparsity(z, rho=rho_target)

            total_loss = (
                w_mse * mse_loss +
                w_mae * mae_loss +
                w_ssim * ssim_loss +
                w_edge * edge_loss +
                w_l1_sparsity * l1_sparsity +
                w_kl_sparsity * kl_sparse
            )

            bs = imgs.size(0)
            sums_val["total"] += total_loss.item() * bs
            sums_val["mse"]   += mse_loss.item()     * bs
            sums_val["mae"]   += mae_loss.item()     * bs
            sums_val["ssim"]  += ssim_loss.item()    * bs
            sums_val["edge"]  += edge_loss.item()    * bs
            sums_val["l1_sparse"] += l1_sparsity.item() * bs
            sums_val["kl_sparse"] += kl_sparse.item()   * bs
            count_val += bs

    for k in loss_names:
        val_hist[k].append(sums_val[k] / max(1, count_val))

    scheduler.step(val_hist["total"][-1])

    print(
        f"Epoch {epoch:3d} | "
        f"Train Tot {train_hist['total'][-1]:.4f} | Val Tot {val_hist['total'][-1]:.4f} | "
        f"Val MSE {val_hist['mse'][-1]:.4f} | Val SSIMloss {val_hist['ssim'][-1]:.4f}"
    )

    # Save best
    if val_hist["total"][-1] < best_val:
        best_val = val_hist["total"][-1]
        best_epoch = epoch
        torch.save(encoder.state_dict(), os.path.join(models_dir, "encoder_best.pth"))
        torch.save(decoder.state_dict(), os.path.join(models_dir, "decoder_best.pth"))

# ----------------------------- Plots ------------------------------------------
# Individual plots for each loss (train vs val)
for k in loss_names:
    plt.figure(figsize=(8, 5))
    plt.plot(train_hist[k], label=f"Train {k}")
    plt.plot(val_hist[k], label=f"Val {k}")
    plt.title(f"{k.capitalize()} Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f"{k}_loss.png"))
    plt.close()

# Combined panel
cols = 3
rows = int(math.ceil(len(loss_names) / cols))
plt.figure(figsize=(cols*5, rows*3.5))
for i, k in enumerate(loss_names, 1):
    plt.subplot(rows, cols, i)
    plt.plot(train_hist[k], label=f"Train {k}")
    plt.plot(val_hist[k], label=f"Val {k}")
    plt.title(k)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True)
    if i == 1:
        plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(models_dir, "all_losses_panel.png"))
plt.close()

# ----------------------------- Final Evaluation & Summary ---------------------
# Load best checkpoint for reporting
encoder.load_state_dict(torch.load(os.path.join(models_dir, "encoder_best.pth"), map_location=device))
decoder.load_state_dict(torch.load(os.path.join(models_dir, "decoder_best.pth"), map_location=device))
encoder.eval(); decoder.eval()

val_psnr, val_ssim = evaluate_metrics(val_loader, encoder, decoder, device)

print("\n===== Training Summary =====")
print(f"Best validation epoch: {best_epoch}")
print(f"Best validation total loss: {best_val:.6f}")
print("Last-epoch (Train):")
for k in loss_names:
    print(f"  {k:>10s}: {train_hist[k][-1]:.6f}")
print("Last-epoch (Val):")
for k in loss_names:
    print(f"  {k:>10s}: {val_hist[k][-1]:.6f}")
print(f"Validation PSNR: {val_psnr:.2f} dB")
print(f"Validation SSIM: {val_ssim:.4f}")
print(f"Latent dim: {latent_dim}, Total params: {num_params:,}")
print(f"Loss weights -> MSE:{w_mse}, MAE:{w_mae}, SSIM:{w_ssim}, EDGE:{w_edge}, L1:{w_l1_sparsity}, KL:{w_kl_sparsity} (rho={rho_target})")

# ----------------------------- Inference Demo ---------------------------------
def denormalize(tensor):
    # convert [-1,1] -> [0,1] for visualization
    return (tensor + 1.0) * 0.5

# Example single-image inference (optional)
image_path = r"C:/Users/PROMOTECH/OneDrive/Documents/DB/DATABASE/test/tst.jpg"
if os.path.exists(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_dim, input_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = encoder(img_tensor).squeeze().detach().cpu().numpy()
        recon_img = decoder(torch.tensor(latent, device=device).unsqueeze(0)).squeeze().detach().cpu()
        recon_img = denormalize(recon_img).permute(1, 2, 0).numpy().clip(0,1)

    Path("latents").mkdir(exist_ok=True)
    with open("latents/latent_vector.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(latent.flatten())

    print("Saved latent vector to latents/latent_vector.csv")

    plt.figure(figsize=(6,3))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed")
    plt.imshow(recon_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "reconstruction_example.png"))
    plt.show()
else:
    print(f"[Info] Skipped single-image inference; file not found: {image_path}")
