import os
import re
import cv2
import json
import numpy as np
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt


def load_image(image_path):
    """Load an image with OpenCV, keeping channel info."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    # Convert to RGB if it has 3 channels in BGR order
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_json(filepath):
    """Try loading a JSON file and return its data, or None if failed."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"❌ Error loading {os.path.basename(filepath)}: {e}")
        return None

def hsv_to_rgb(h, s, v):
    """
    Convert HSV to RGB (all 0-1 range).
    """
    hsv_pixel = np.uint8([[[int(h*179), int(s*255), int(v*255)]]])
    rgb = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0] / 255
    return tuple(rgb)

def generate_distinct_colors(num_classes):
    """
    Generate maximally distinct RGB colors by evenly spacing hues.
    Returns a list of RGB tuples in 0-1 range.
    """
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        saturation = 1.0
        value = 1.0
        color = hsv_to_rgb(hue, saturation, value)
        colors.append(color)
    return colors

def save_json(json_object, filepath):
    with open(filepath, 'w') as f:
        json.dump(json_object, f, indent=4)

def save_mask(mask, output_path):
    """Save the segmentation mask."""
    plt.imsave(output_path, mask, cmap='jet')
    print(f"Mask saved to {output_path}")

def resolve_files(inputs, valid_exts):
    if isinstance(inputs, str) and os.path.isdir(inputs):
        return sorted([
            os.path.join(inputs, f)
            for f in os.listdir(inputs)
            if Path(f).suffix.lower() in valid_exts
        ])
    elif isinstance(inputs, list):
        return sorted([f for f in inputs if Path(f).suffix.lower() in valid_exts])
    else:
        raise ValueError("Inputs must be a directory or list of file paths.")

def extract_well_site(filename):
    """
    Extracts the Plate_Site pattern (e.g., 'A6_s2') from a filename.
    """
    match = re.search(r'([A-Z]\d+_s\d+)', filename)
    return match.group(1) if match else None

def get_image_mask_pairs_from_folder(image_folder, mask_folder, verbose=False):
    """
    Scans the image and mask folders and returns matching image/mask path pairs.

    Matches masks based on 'Plate_Site' naming (e.g., 'A6_s2').
    Supports multiple masks per image.

    Returns:
        image_paths (list of str)
        mask_paths (list of str)  # Each image may have multiple masks.
    """
    valid_exists = ('.png', '.tif', '.tiff', '.json')

    # --- Step 1: Index masks ---
    mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(valid_exists)]
    mask_index = {}

    for mask_file in mask_files:
        plate_site = extract_well_site(mask_file)
        if plate_site:
            key = plate_site.lower()
            mask_index.setdefault(key, []).append(mask_file)

    if verbose:
        total_masks = sum(len(v) for v in mask_index.values())
        print(f"Indexed {total_masks} masks across {len(mask_index)} Plate_Sites.")

    # --- Step 2: Process images ---
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_exists)]
    image_paths = []
    mask_paths = []

    for img_file in image_files:
        plate_site = extract_well_site(img_file)
        if not plate_site:
            if verbose:
                print(f"[SKIP] Couldn't extract Plate_Site from {img_file}")
            continue

        masks_for_site = mask_index.get(plate_site.lower(), [])

        if masks_for_site:
            img_path = os.path.join(image_folder, img_file)
            full_mask_paths = [os.path.join(mask_folder, m) for m in masks_for_site
                               if os.path.isfile(os.path.join(mask_folder, m))]

            if full_mask_paths:
                image_paths.append(img_path)
                mask_paths.append(full_mask_paths)
                if verbose:
                    print(f"[MATCH] {img_file} ↔ {len(full_mask_paths)} mask(s).")
            else:
                if verbose:
                    print(f"[NO FILES] No valid mask files found on disk for {img_file}.")
        else:
            if verbose:
                print(f"[NO MASK] No masks found for {img_file}.")

    return image_paths, mask_paths

# DNN Training
def load_images_and_masks_from_folder(folder, verbose=True):
    """
    Loads images and matching _mask files from a folder for DNN training.

    Returns:
        images (list of np.ndarray)
        masks (list of np.ndarray)
    """

    image_files = [f for f in os.listdir(folder)
                   if f.endswith('.png') and not f.endswith('_mask.png')]

    if verbose:
        print(f"[INFO] Found {len(image_files)} base images in {folder}")

    missing_masks = []
    for img_file in image_files:
        expected_mask = img_file.replace('.png', '_mask.png')
        if not os.path.exists(os.path.join(folder, expected_mask)):
            missing_masks.append(expected_mask)

    if missing_masks:
        raise ValueError(f"Missing masks for images: {missing_masks}")

    images = [imread(os.path.join(folder, f)) for f in image_files]
    masks = [imread(os.path.join(folder, f.replace('.png', '_mask.png')))
             for f in image_files]

    return images, masks

# ---- Test code ----
if __name__ == "__main__":
    import pprint

    # --- Set your test paths here ---
    image_folder = "../data/images"
    mask_folder = "../data/annotations"

    print("\n--- Testing extract_plate_site ---")
    test_filenames = [
        "Araceli_A6_s2_w1_z0_1020e47f-73ff-427f-b5aa-44d2915e9068.tiff",
        "A6_s2.json",
        "invalid_file_name.tif"
    ]
    for fname in test_filenames:
        result = extract_well_site(fname)
        print(f"{fname} -> {result}")

    print("\n--- Testing get_image_mask_pairs_from_folder ---")
    image_paths, mask_paths = get_image_mask_pairs_from_folder(image_folder, mask_folder, verbose=True)
    print(f"\nMatched {len(image_paths)} image-mask pairs.")
    for img, masks in zip(image_paths, mask_paths):
        print(f"{os.path.basename(img)} ↔ {len(masks)} mask(s): {[os.path.basename(m) for m in masks]}")

