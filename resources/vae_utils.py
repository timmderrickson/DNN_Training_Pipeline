import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from resources import helper_functions as hf
from torch.utils.data import DataLoader, TensorDataset


# ---- VAE Architecture ---- #
class VAE(nn.Module):
    def __init__(self, img_channels=1, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # 512 → 256
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),            # 256 → 128
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),           # 128 → 64
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          # 64 → 32
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(256 * 32 * 32, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 256 * 32 * 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64 → 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 128 → 256
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 256 → 512
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 32, 32)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ---- Loss ---- #
def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


# ---- Dataset Utility ---- #
def get_tensor_dataloader(image_list, batch_size=8, verbose=False, tile_size=(512, 512), overlap=0):
    """
    Converts a list of images into a PyTorch DataLoader, tiling large images automatically.

    Args:
        image_list (list): List of 2D np.ndarray images.
        batch_size (int): Batch size for DataLoader.
        verbose (bool): Whether to print debug info.
        tile_size (tuple): Size of each tile (default 512x512).
        overlap (int): Overlap between tiles in pixels.

    Returns:
        DataLoader for torch tensors of shape (N, 1, H, W)
    """
    tensors = []
    for idx, img in enumerate(image_list):
        norm_img = hf.normalize_image(img, verbose=verbose)

        # Auto-tile if image is larger than tile size
        if norm_img.shape[0] > tile_size[0] or norm_img.shape[1] > tile_size[1]:
            tiles, coords, padded_shape = hf.tile_image(norm_img, tile_size=tile_size, overlap=overlap)
            if verbose:
                print(f"[INFO] Image {idx} tiled into {len(tiles)} patches.")
            for tile in tiles:
                tile_tensor = torch.from_numpy(tile).unsqueeze(0)  # (H, W) → (1, H, W)
                tensors.append(tile_tensor)
        else:
            img_tensor = torch.from_numpy(norm_img).unsqueeze(0)
            tensors.append(img_tensor)

    tensor_stack = torch.stack(tensors)
    return DataLoader(TensorDataset(tensor_stack), batch_size=batch_size, shuffle=True)


# ---- Training ---- #
def train_vae(image_list, latent_dim=128, batch_size=8, lr=1e-3, num_epochs=20, save_path='vae_model.pth', verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataloader = get_tensor_dataloader(image_list, batch_size)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = batch[0].to(device)
            optimizer.zero_grad()

            recon_imgs, mu, logvar = model(imgs)
            loss = vae_loss_function(recon_imgs, imgs, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch+1}: Loss = {running_loss:.2f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    if verbose:
        print(f"✅ VAE model saved to {save_path}")

    return model


# ---- Inference ---- #
def generate_images_from_vae(model_path, num_images=20, latent_dim=128, image_shape=(512, 512)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        generated = model.decode(z).cpu().numpy()

    generated = np.clip(generated, 0, 1)
    generated_imgs = (generated * 255).astype(np.uint8)
    generated_imgs = [np.squeeze(img) for img in generated_imgs]  # remove channel dim

    return generated_imgs

def visualize_vae_outputs(image_list, title="VAE Generated Samples", max_display=10):
    """
    Display a row of VAE-generated grayscale images.

    Parameters:
        image_list (list of np.ndarray): List of 2D grayscale images.
        title (str): Title for the plot.
        max_display (int): Max number of images to show.
    """
    num_images = min(len(image_list), max_display)
    plt.figure(figsize=(3 * num_images, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image_list[i], cmap='gray')
        plt.title(f"Sample {i+1}")
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# ---- MAIN ENTRY POINT ---- #
if __name__ == "__main__":
    import cv2
    import glob

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "..", "data", "images")
    model_save_path = os.path.join(script_dir, "..", "models", "trained_models", "models", "vae_model.pth")


    # 1️⃣ Load images using your helper
    image_paths = glob.glob(os.path.join(input_dir, "*.tiff"))
    print(f"[INFO] Found {len(image_paths)} .tiff images.")

    image_list = []

    for p in image_paths:
        try:
            img = hf.load_image(p)
            print(img.shape)
            if len(img.shape) == 2:  # Grayscale
                image_list.append(img)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                image_list.append(gray)
            else:
                print(f"[SKIP] Unsupported shape {img.shape} for {p}")
        except Exception as e:
            print(f"[ERROR] Could not load image {p}: {e}")

    print(f"[INFO] Loaded {len(image_list)} valid grayscale images.")

    if len(image_list) == 0:
        raise RuntimeError("No valid grayscale images loaded — check your image folder and formats.")

    # 2️⃣ Train VAE
    model = train_vae(image_list=image_list,
                      latent_dim=128,
                      batch_size=8,
                      lr=1e-3,
                      num_epochs=20,
                      save_path=model_save_path)

    # 3️⃣ Generate new images
    synthetic_images = generate_images_from_vae(model_path=model_save_path,
                                                num_images=10,
                                                latent_dim=128)

    # 4️⃣ Visualize
    visualize_vae_outputs(synthetic_images, title="Synthetic VAE Samples")
