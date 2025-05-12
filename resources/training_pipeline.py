import os
import cv2
import random
import shutil
import numpy as np
import pandas as pd
import albumentations as A
from datetime import datetime
import matplotlib.pyplot as plt

from models import model_functions as mf
from resources import helper_functions as hf
from resources import json_conversion_tools as jc
from resources import polygon_json_visualizations as viz


# 1 Data Preparation
def find_matching_images(json_path, image_folder, verbose=False):
    """
    For a given JSON annotation path, find all matching TIFF images in image_folder.
    Matches based on the well and site (e.g., 'A6_s2'), accounting for optional 'Araceli_' prefix.

    Parameters:
        json_path (str): Full path to the JSON annotation file (e.g., '../data/annotations/A6_s2.json').
        image_folder (str): Folder containing TIFF images.
        verbose (bool): If True, print found matches.

    Returns:
        list of str: Full paths to matching image files.
    """
    # Extract filename from path → 'A6_s2.json'
    json_filename = os.path.basename(json_path)

    # Remove .json extension → 'A6_s2'
    well_site = os.path.splitext(json_filename)[0]

    # List all TIFF files in the image folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.tiff', '.tif'))]

    # Match filenames that start with either 'A6_s2_' or 'Araceli_A6_s2_'
    matching_images = []
    for f in image_files:
        if f.startswith(well_site + '_') or f.startswith('Araceli_' + well_site + '_'):
            matching_images.append(os.path.join(image_folder, f))

    if verbose:
        print(f"JSON: {json_filename}")
        if matching_images:
            print(f"  Found {len(matching_images)} matching images:")
            for img in matching_images:
                print(f"    {os.path.basename(img)}")
        else:
            print("  No matching images found.")

    return matching_images

def validate_instance_mask(instance_mask, json_path, verbose=False):
    """
    Validates the instance mask created from a JSON annotation.

    Parameters:
        instance_mask (np.ndarray): Instance mask where each object has a unique ID.
        json_path (str): Path to the JSON file (for reporting).
        verbose (bool): If True, prints detailed validation info.

    Returns:
        bool: True if the mask passes validation, False otherwise.
    """

    unique_ids = np.unique(instance_mask)
    instance_ids = unique_ids[unique_ids > 0]
    num_instances = len(instance_ids)

    valid = True

    if num_instances == 0:
        if verbose:
            print(f"[WARNING] {json_path}: No objects found in mask.")
        valid = False

    if verbose and valid:
        print(f"[INFO] {json_path}: Validation passed with {num_instances} instances (IDs: {instance_ids}).")

    return valid

def prepare_training_data(gt_json_folder, image_folder, output_image_folder, output_mask_folder,
                          image_shape, classes_to_include=None, verbose=False, preview=False):
    """
    Converts JSON annotations to instance masks and copies matching images.
    Organizes data into Cellpose training format.

    Images are only copied if they don't already exist in the output folder.
    Masks are always overwritten to ensure latest version.

    Parameters:
        preview (bool): If True, visualize the annotation overlay for one image per JSON.
    """

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    gt_json_files = [f for f in os.listdir(gt_json_folder) if f.endswith('.json')]

    for gt_json_file in gt_json_files:
        gt_json_path = os.path.join(gt_json_folder, gt_json_file)

        if verbose:
            print(f"\nProcessing JSON: {gt_json_file}")

        # 1️⃣ Convert JSON to instance mask
        instance_mask = jc.load_polygon_mask_for_cellpose(
            json_path=gt_json_path,
            image_shape=image_shape,
            classes_to_include=classes_to_include,
            verbose=verbose
        )

        # 2️⃣ Validate the mask
        is_valid = validate_instance_mask(instance_mask, gt_json_path, verbose=verbose)
        if not is_valid:
            if verbose:
                print(f"[SKIP] {gt_json_file} — invalid mask.")
            continue

        # 3️⃣ Find matching images
        matching_images = find_matching_images(gt_json_path, image_folder, verbose=verbose)
        if len(matching_images) == 0:
            if verbose:
                print(f"[SKIP] {gt_json_file} — no matching images found.")
            continue

        # 4️⃣ Copy images to output folder (skip if already exists)
        for img_path in matching_images:
            img_filename = os.path.basename(img_path)
            output_img_path = os.path.join(output_image_folder, img_filename)

            if not os.path.exists(output_img_path):
                shutil.copy2(img_path, output_image_folder)
                if verbose:
                    print(f"[COPY] Image copied: {img_filename}")
            else:
                if verbose:
                    print(f"[SKIP] Image already exists: {img_filename}")

        # 5️⃣ Save (or overwrite) instance mask
        well_site = os.path.splitext(gt_json_file)[0]  # e.g., 'A1_s1'
        mask_suffix = "instance_mask"
        mask_filename = f"{well_site}_{mask_suffix}.png"
        mask_output_path = os.path.join(output_mask_folder, mask_filename)

        cv2.imwrite(mask_output_path, instance_mask)
        if verbose:
            print(f"[SAVE] Mask saved (overwritten if existed): {mask_filename}")

        # 6️⃣ OPTIONAL: Preview annotation overlay
        if preview:
            try:
                # Pick the first matching image for visualization
                image_to_visualize = matching_images[0]
                gt_to_visualize = hf.load_json(gt_json_path)
                # Visualize with the original JSON annotations
                viz.visualize_masks(
                    image_path=image_to_visualize,
                    pred_json=gt_to_visualize,
                    classes_to_include=classes_to_include,
                    model="annotation"
                )

            except Exception as e:
                print(f"[WARNING] Visualization failed for {gt_json_file}: {e}")

    print(f"\n✅ Data preparation complete. JSONs processed: {len(gt_json_files)}")

    return output_image_folder, output_mask_folder

# 2 Optional: Augmentation
def get_augmentation_pipeline(config):
    """
    Builds an Albumentations pipeline based on the augmentation_config.

    Parameters:
        config (dict): Augmentation settings.

    Returns:
        albumentations.Compose: The augmentation pipeline.
    """

    transforms = []

    # Horizontal flip
    if config.get("horizontal_flip", 0) > 0:
        transforms.append(A.HorizontalFlip(p=config["horizontal_flip"]))

    # Vertical flip
    if config.get("vertical_flip", 0) > 0:
        transforms.append(A.VerticalFlip(p=config["vertical_flip"]))

    # Random 90-degree rotation
    if config.get("random_rotation_90", 0) > 0:
        transforms.append(A.RandomRotate90(p=config["random_rotation_90"]))

    # Scaling
    scaling_range = config.get("scaling_range", (1.0, 1.0))
    if scaling_range != (1.0, 1.0):
        transforms.append(
            A.RandomScale(scale_limit=(scaling_range[0]-1, scaling_range[1]-1), p=0.5)
        )

    # Brightness/contrast
    brightness = config.get("brightness_range", (1.0, 1.0))
    contrast = config.get("contrast_range", (1.0, 1.0))
    if brightness != (1.0, 1.0) or contrast != (1.0, 1.0):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=(brightness[0]-1, brightness[1]-1),
                contrast_limit=(contrast[0]-1, contrast[1]-1),
                p=0.5
            )
        )

    # Optional blur
    if config.get("apply_blur", False):
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=0.3))

    # Optional translation (shift)
    translation_pixels = config.get("translation_pixels", 0)
    if translation_pixels > 0:
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit_x=translation_pixels / config.get("image_shape", (3000, 3000))[1],
                shift_limit_y=translation_pixels / config.get("image_shape", (3000, 3000))[0],
                scale_limit=0,
                rotate_limit=0,
                p=0.5
            )
        )

    # Always resize to original size (in case scaling/translation changes size)
    transforms.append(
        A.LongestMaxSize(max_size=max(config.get("image_shape", (3000, 3000))), always_apply=True)
    )
    transforms.append(
        A.PadIfNeeded(
            min_height=config.get("image_shape", (3000, 3000))[0],
            min_width=config.get("image_shape", (3000, 3000))[1],
            border_mode=0,  # constant padding
            value=0,
            mask_value=0,
            always_apply=True
        )
    )

    return A.Compose(
        transforms,
        additional_targets={'mask': 'mask'}  # Ensure mask gets same transforms
    )

def create_synthetic_image_and_mask(image_shape=(512, 512), num_shapes=5):
    """
    Generates a synthetic grayscale image and an instance mask
    with multiple non-uniform shapes.

    Parameters:
        image_shape (tuple): Size of the generated image and mask.
        num_shapes (int): Number of random shapes to include.

    Returns:
        image (np.ndarray): Grayscale image.
        mask (np.ndarray): Instance mask with unique IDs for each shape.
    """

    image = np.zeros(image_shape, dtype=np.uint8)
    mask = np.zeros(image_shape, dtype=np.uint16)

    for i in range(1, num_shapes + 1):
        shape_type = random.choice(["ellipse", "polygon"])

        if shape_type == "ellipse":
            center = (
                random.randint(50, image_shape[1] - 50),
                random.randint(50, image_shape[0] - 50)
            )
            axes = (
                random.randint(20, 60),
                random.randint(20, 60)
            )
            angle = random.randint(0, 180)

            cv2.ellipse(image, center, axes, angle, 0, 360, color=150 + 10 * i, thickness=-1)
            cv2.ellipse(mask, center, axes, angle, 0, 360, color=i, thickness=-1)

        else:  # polygon
            num_pts = random.randint(4, 8)
            pts = []
            for _ in range(num_pts):
                x = random.randint(20, image_shape[1] - 20)
                y = random.randint(20, image_shape[0] - 20)
                pts.append([x, y])
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(image, [pts], color=100 + 10 * i)
            cv2.fillPoly(mask, [pts], color=i)

    return image, mask

def visualize_augmentation_samples(aug_pipeline, image, mask, num_samples=5):
    """
    Applies augmentations and visualizes original + augmented image/mask pairs.

    Parameters:
        aug_pipeline (albumentations.Compose): The augmentation pipeline.
        image (np.ndarray): Grayscale image.
        mask (np.ndarray): Instance mask.
        num_samples (int): Number of augmented samples to generate.
    """

    plt.figure(figsize=(4 * (num_samples + 1), 5))

    # Show original
    plt.subplot(1, num_samples + 1, 1)
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.4)
    plt.title("Original")
    plt.axis('off')

    # Show augmentations
    for i in range(num_samples):
        augmented = aug_pipeline(image=image, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        plt.subplot(1, num_samples + 1, i + 2)
        plt.imshow(aug_img, cmap='gray')
        plt.imshow(aug_mask, cmap='jet', alpha=0.4)
        plt.title(f"Aug {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def extract_image_mask_pairs(augmented_data):
    """
    Extracts all augmented images and masks from the returned augmented data object.

    Parameters:
        augmented_data (list of dict): Output from apply_augmentation_batch.

    Returns:
        images (list of np.ndarray): List of augmented images.
        masks (list of np.ndarray): List of augmented masks.
    """
    images = []
    masks = []

    for entry in augmented_data:
        images.append(entry['aug_image'])
        masks.append(entry['aug_mask'])

    return images, masks

def prepare_augmented_training_data(image_folder, mask_folder, augmentation_config, num_augments=5, verbose=True):
    """
    Loads image/mask pairs, applies augmentations, and returns augmented data ready for training.
    Supports multiple instance masks per image (they are merged safely before augmentation).
    """

    # ----------------------------------------
    # Step 1: Load image/mask paths
    # ----------------------------------------
    image_paths, mask_paths_list = hf.get_image_mask_pairs_from_folder(image_folder, mask_folder, verbose=verbose)

    if len(image_paths) == 0:
        raise ValueError("No image/mask pairs found.")

    if verbose:
        print(f"\nFound {len(image_paths)} valid image/mask pairs.")

    # ----------------------------------------
    # Step 2: Build augmentation pipeline
    # ----------------------------------------
    aug_pipeline = get_augmentation_pipeline(augmentation_config)

    if verbose:
        print("\nAugmentation pipeline created.")

    # ----------------------------------------
    # Step 3: Apply augmentations in batch
    # ----------------------------------------
    augmented_data = []

    for idx, (img_path, mask_paths) in enumerate(zip(image_paths, mask_paths_list)):
        if verbose:
            print(f"\n[{idx + 1}/{len(image_paths)}] Processing {os.path.basename(img_path)}")

        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # ----------------------------------------
        # Load and safely merge all instance masks
        # ----------------------------------------
        masks = []
        for mask_file in mask_paths:
            m = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            if m is None:
                raise ValueError(f"Failed to load mask file: {mask_file}")
            masks.append(m)

        # Start with empty combined mask
        combined_mask = np.zeros_like(masks[0], dtype=np.uint16)
        current_max_id = 0

        for m in masks:
            mask_instances = np.unique(m)
            mask_instances = mask_instances[mask_instances > 0]  # Exclude background (0)

            if len(mask_instances) > 0:
                # Offset the instance IDs to prevent overlaps
                offset_mask = np.where(m > 0, m + current_max_id, 0)
                combined_mask = np.maximum(combined_mask, offset_mask)
                current_max_id = combined_mask.max()

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # ----------------------------------------
        # Augmentations
        # ----------------------------------------
        for aug_idx in range(1, num_augments + 1):
            augmented = aug_pipeline(image=image, mask=combined_mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']

            augmented_data.append({
                "aug_image": aug_img,
                "aug_mask": aug_mask,
                "original_image_path": img_path,
                "original_mask_paths": mask_paths,
                "augmentation_idx": aug_idx
            })

    if verbose:
        print(f"\n✅ Augmentation complete. {len(augmented_data)} augmented samples created.")

    return augmented_data

# 3 Train/Test Split
def split_augmented_data(augmented_data, train_ratio=0.8, random_seed=42, verbose=True, save_split=True, save_dir=None,
                         return_data=False):
    """
    Splits augmented_data into training and validation sets.

    Parameters:
        augmented_data (list of dict): Output from prepare_augmented_training_data.
        train_ratio (float): Fraction of data to use for training.
        random_seed (int): Seed for reproducible shuffling.
        verbose (bool): Whether to print split counts.
        save_split (bool): Whether to save the split datasets to disk.
        save_dir (str or None): Directory to save the train/val split (default: './training_data/augmented_data_split').
        return_data (bool): Whether to return in-memory train_data and val_data lists.

    Returns:
        train_data (list of dict), val_data (list of dict) if return_data is True, else None.
    """

    total_samples = len(augmented_data)

    # Shuffle indices
    indices = list(range(total_samples))
    random.Random(random_seed).shuffle(indices)

    # Shuffle the augmented data list
    shuffled_data = [augmented_data[i] for i in indices]

    # Compute split sizes
    num_train = int(total_samples * train_ratio)

    train_data = shuffled_data[:num_train]
    val_data = shuffled_data[num_train:]

    if verbose:
        print(f"\n✅ Total samples: {total_samples}")
        print(f"Training set: {len(train_data)} samples")
        print(f"Validation set: {len(val_data)} samples")

    # Optional save
    if save_split:
        if save_dir is None:
            save_dir = './training_data/augmented_data_split'

        # IMPORTANT: Images and masks will go into the SAME folder!
        train_img_dir = os.path.join(save_dir, 'train')
        val_img_dir = os.path.join(save_dir, 'val')

        for d in [train_img_dir, val_img_dir]:
            os.makedirs(d, exist_ok=True)

        def save_data_split(data_split, img_dir, split_name):
            for entry in data_split:
                aug_idx = entry['augmentation_idx']
                base_name = os.path.splitext(os.path.basename(entry['original_image_path']))[0]
                img_filename = f"{base_name}_aug_{aug_idx}.png"
                mask_filename = f"{base_name}_aug_{aug_idx}_mask.png"

                img_path = os.path.join(img_dir, img_filename)
                mask_path = os.path.join(img_dir, mask_filename)  # BOTH go into img_dir!

                cv2.imwrite(img_path, entry['aug_image'])
                cv2.imwrite(mask_path, entry['aug_mask'])

                if verbose:
                    print(f"[SAVED {split_name.upper()}] {img_filename} and {mask_filename}")

        if verbose:
            print("\nSaving training data...")
        save_data_split(train_data, train_img_dir, 'train')

        if verbose:
            print("\nSaving validation data...")
        save_data_split(val_data, val_img_dir, 'val')

        if verbose:
            print("\n✅ Saved split datasets to:", save_dir)

    if return_data:
        return train_data, val_data
    else:
        return None, None

# 4 Training
def run_training_from_split(split_dir, net='CPnetV2', save_path='training/trained_cellpose_model',
                            n_epochs=500, learning_rate=0.2, weight_decay=1e-5, batch_size=8, gpu=True, verbose=True):
    """
    Prepares training and validation directories and starts Cellpose training.
    Note: pretrained_model, channels, and gpu are ignored in Cellpose 4.x training.
    """

    train_images_dir = os.path.abspath(os.path.join(split_dir, 'train')).replace("\\", "/")
    val_images_dir = os.path.abspath(os.path.join(split_dir, 'val')).replace("\\", "/")

    if verbose:
        print(f"Training directory:    {train_images_dir}")
        print(f"Validation directory: {val_images_dir}")
        print("Note: pretrained_model, channels, and gpu arguments are ignored in Cellpose 4.x training.")

    mf.train_cellpose_model(
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        save_path=save_path,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        net=net,
        gpu=gpu  # Use CPU to avoid GPU errors in test
    )

def plot_cellpose_training_metrics(save_path='training/trained_cellpose_model'):
    """
    Loads and plots Cellpose training metrics from the save_path.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    metrics_path = os.path.join(save_path, 'metrics.npy')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No metrics.npy file found in {save_path}")

    metrics = np.load(metrics_path, allow_pickle=True).item()

    train_loss = metrics.get('train_loss', [])
    val_loss = metrics.get('val_loss', [])

    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    if len(val_loss) > 0:
        plt.plot(epochs, val_loss, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Cellpose Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()


# 5 Model Saving and Logging
def save_final_checkpoint(model_save_path='training/trained_cellpose_model', experiment_dir="trained_models"):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_name = f"cellpose_model_{timestamp}.pth"

    final_model_path = os.path.join(model_save_path, 'cellpose_model.pth')
    if os.path.exists(final_model_path):
        os.makedirs(experiment_dir, exist_ok=True)
        shutil.copy2(final_model_path, os.path.join(experiment_dir, save_name))
        print(f"✅ Final model checkpoint saved as {save_name}")
    else:
        print(f"❌ No final model found at {final_model_path}")

def save_metrics_as_csv(metrics_path='training/trained_cellpose_model/metrics.npy',
                        output_csv_path='training/trained_cellpose_model/metrics.csv'):
    metrics = np.load(metrics_path, allow_pickle=True).item()
    df = pd.DataFrame({
        'epoch': list(range(1, len(metrics['train_loss']) + 1)),
        'train_loss': metrics['train_loss'],
        'val_loss': metrics.get('val_loss', [None]*len(metrics['train_loss']))
    })
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Metrics saved to {output_csv_path}")

def save_training_summary(n_epochs, learning_rate, weight_decay, batch_size, train_loss, val_loss=None,
                          pretrained_model='CPSAM', save_path='training/trained_cellpose_model', additional_notes=None):
    """
    Saves a human-readable training summary text file to the save_path.

    Parameters:
        n_epochs (int): Number of epochs trained.
        learning_rate (float): Learning rate used.
        weight_decay (float): Weight decay used.
        batch_size (int): Batch size.
        train_loss (list or np.ndarray): Training loss history.
        val_loss (list or np.ndarray or None): Validation loss history.
        pretrained_model (str): Pretrained model used.
        save_path (str): Directory where the summary will be saved.
        additional_notes (str or None): Any extra notes to include.
    """

    # Make sure save directory exists
    os.makedirs(save_path, exist_ok=True)

    summary_lines = []
    summary_lines.append("===== CELLPOSE TRAINING SUMMARY =====")
    summary_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    summary_lines.append(f"Pretrained Model: {pretrained_model}")
    summary_lines.append(f"Epochs: {n_epochs}")
    summary_lines.append(f"Learning Rate: {learning_rate}")
    summary_lines.append(f"Weight Decay: {weight_decay}")
    summary_lines.append(f"Batch Size: {batch_size}")
    summary_lines.append("")

    # Final losses
    if train_loss is not None and len(train_loss) > 0:
        summary_lines.append(f"Final Training Loss: {train_loss[-1]:.6f}")
    else:
        summary_lines.append("Final Training Loss: N/A")

    if val_loss is not None and len(val_loss) > 0:
        summary_lines.append(f"Final Validation Loss: {val_loss[-1]:.6f}")
    else:
        summary_lines.append("Final Validation Loss: N/A")

    summary_lines.append("")
    summary_lines.append("Notes:")
    if additional_notes:
        summary_lines.append(additional_notes)
    else:
        summary_lines.append("None")

    summary_file = os.path.join(save_path, "training_summary.txt")
    with open(summary_file, 'w') as f:
        for line in summary_lines:
            f.write(line + "\n")

    print(f"✅ Training summary saved to {summary_file}")


if __name__ == "__main__":

    # === CLEAN TEST FOLDERS ===
    import json
    test_root = "../testing/training_pipeline"
    gt_json_folder = os.path.join(test_root, "annotations")
    image_folder = os.path.join(test_root, "images")
    output_image_folder = os.path.join(test_root, "prepared_images")
    output_mask_folder = os.path.join(test_root, "prepared_masks")
    split_save_dir = os.path.join(test_root, "augmented_data_split")

    for d in [gt_json_folder, image_folder, output_image_folder, output_mask_folder, split_save_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # === STEP 0: CREATE SYNTHETIC DATA ===

    num_samples = 3
    image_shape = (512, 512)

    for i in range(num_samples):
        img, mask = create_synthetic_image_and_mask(image_shape=image_shape, num_shapes=5)

        # Save image
        well_site = f"A{i+1}_s1"
        image_filename = f"Araceli_{well_site}_z0_fakeid.tiff"
        cv2.imwrite(os.path.join(image_folder, image_filename), img)

        # Create annotation JSON
        annotations_by_class = {"synthetic_class": []}  # You can name the class anything

        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0]

        for obj_id in instance_ids:
            # Find all pixels belonging to this object
            ys, xs = np.where(mask == obj_id)
            if len(xs) < 3:
                continue  # Need at least 3 points for a polygon

            # Just use the convex hull as a placeholder polygon
            pts = np.stack([xs, ys], axis=1)
            hull = cv2.convexHull(pts)

            # Flatten the hull to a list of points
            points = hull.squeeze().tolist()
            if isinstance(points[0], list):
                points_flat = [tuple(pt) for pt in points]
            else:
                points_flat = [tuple(points)]

            annotations_by_class["synthetic_class"].append({
                "type": "POLYGON",
                "id": int(obj_id),
                "points": points_flat
            })

        json_filename = f"{well_site}.json"
        hf.save_json(annotations_by_class, os.path.join(gt_json_folder, json_filename))

    print("\n✅ Synthetic data created.")

    # === STEP 1: PREPARE TRAINING DATA ===

    prepared_image_folder, prepared_mask_folder = prepare_training_data(
        gt_json_folder=gt_json_folder,
        image_folder=image_folder,
        output_image_folder=output_image_folder,
        output_mask_folder=output_mask_folder,
        image_shape=image_shape,
        classes_to_include=None,
        verbose=True,
        preview=False
    )

    # === STEP 2: AUGMENTATION ===

    augmentation_config = {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.5,
        "random_rotation_90": 0.5,
        "scaling_range": (0.9, 1.1),
        "brightness_range": (0.9, 1.1),
        "contrast_range": (0.9, 1.1),
        "apply_blur": False,
        "translation_pixels": 10,
        "image_shape": image_shape
    }

    num_augments = 2  # For fast testing

    augmented_data = prepare_augmented_training_data(
        image_folder=prepared_image_folder,
        mask_folder=prepared_mask_folder,
        augmentation_config=augmentation_config,
        num_augments=num_augments,
        verbose=True
    )

    print(f"\n✅ Augmented samples generated: {len(augmented_data)}")

    # === STEP 3: SPLIT ===

    train_data, val_data = split_augmented_data(
        augmented_data=augmented_data,
        train_ratio=0.8,
        random_seed=42,
        verbose=True,
        save_split=True,
        save_dir=split_save_dir,
        return_data=True
    )

    print(f"\n✅ Train/Val split complete. Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Check folder: {split_save_dir}")
