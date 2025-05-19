import os
import cv2
import glob
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from resources import helper_functions as hf


# Generator Network
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: [B, 100, 1, 1]
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),  # → 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),         # → 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),         # → 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),          # → 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),           # → 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),           # → 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, img_channels, 4, 2, 1),  # → 256x256
            nn.ReLU(True),

            nn.ConvTranspose2d(img_channels, img_channels, 4, 2, 1),  # → 512x512
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)



# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: [B, 1, 512, 512]
            nn.Conv2d(img_channels, 64, 4, 2, 1),    # → [B, 64, 256, 256]
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),             # → [B, 128, 128, 128]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),            # → [B, 256, 64, 64]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),            # → [B, 512, 32, 32]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1),            # → [B, 512, 16, 16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1),            # → [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),                            # → [B, 512 * 8 * 8 = 32768]
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


def train_gan(image_list, latent_dim=100, batch_size=16, num_epochs=100,
              save_dir='models/gan/', device=None, verbose=True):

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    tensors = []
    for img in image_list:
        img = hf.normalize_image(img)
        if img.shape[0] > 512 or img.shape[1] > 512:
            tiles, _, _ = hf.tile_image(img, tile_size=(512, 512))
        else:
            tiles = [img]

        for tile in tiles:
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).float()  # (1, H, W)
            tensors.append(tile_tensor)


    data = torch.stack(tensors).to(device)
    dataset = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    # Initialize models
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optim_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        for real_batch in dataset:
            real_imgs = real_batch[0].to(device)

            # Labels
            real_labels = torch.ones(real_imgs.size(0), 1, device=device)
            fake_labels = torch.zeros(real_imgs.size(0), 1, device=device)

            # Train Discriminator
            z = torch.randn(real_imgs.size(0), latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)

            real_preds = discriminator(real_imgs)
            fake_preds = discriminator(fake_imgs.detach())

            d_loss_real = criterion(real_preds, real_labels)
            d_loss_fake = criterion(fake_preds, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # Train Generator
            z = torch.randn(real_imgs.size(0), latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)
            g_loss = criterion(discriminator(fake_imgs), real_labels)

            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

        if verbose:
            print(f"[{epoch+1}/{num_epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            save_image(fake_imgs[:8], os.path.join(save_dir, f"generated_epoch_{epoch+1}.png"), normalize=True)

    torch.save(generator.state_dict(), os.path.join(save_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator.pth"))
    print(f"✅ Model saved to {save_dir}")
    return generator

def generate_from_gan(model_path, latent_dim=100, num_samples=8, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    z = torch.randn(num_samples, latent_dim, 1, 1).to(device)
    with torch.no_grad():
        samples = generator(z).cpu().numpy()

    images = [np.squeeze((img + 1) / 2 * 255.0).astype(np.uint8) for img in samples]  # [-1,1] → [0,255]
    return images

# --- VISUALIZE OUTPUT ---
def visualize_gan_outputs(image_list, title="GAN Generated Samples"):
    num_images = len(image_list)
    plt.figure(figsize=(3 * num_images, 3))
    for i, img in enumerate(image_list):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"Sample {i+1}")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- CONFIGURATION ---
    input_dir = os.path.join(os.path.dirname(__file__), "..", "data", "images")
    model_save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "trained_gan")
    os.makedirs(model_save_dir, exist_ok=True)

    latent_dim = 100
    num_epochs = 20
    batch_size = 8

    # --- LOAD IMAGES ---
    image_paths = glob.glob(os.path.join(input_dir, "*.tiff"))
    image_list = []

    for p in image_paths:
        try:
            img = hf.load_image(p)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            image_list.append(img)
        except Exception as e:
            print(f"[ERROR] Failed to load image {p}: {e}")

    print(f"[INFO] Loaded {len(image_list)} grayscale images for GAN training.")

    if not image_list:
        raise RuntimeError("No images found — check your input directory and formats.")

    # --- TRAIN GAN ---
    generator = train_gan(
        image_list=image_list,
        latent_dim=latent_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
        save_dir=model_save_dir,
        verbose=True
    )

    # --- GENERATE SAMPLES ---
    model_path = os.path.join(model_save_dir, "generator.pth")
    generated_images = generate_from_gan(
        model_path=model_path,
        latent_dim=latent_dim,
        num_samples=8
    )

    visualize_gan_outputs(generated_images)
