import os
import cv2
import numpy as np
import shutil
from training_pipeline import prepare_augmented_training_data

def create_dummy_image_and_masks(base_dir, img_name="A1_s1_test_image.png"):
    """
    Creates a dummy grayscale image and two instance masks.
    """
    img_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Create dummy image (grayscale)
    img = np.full((100, 100), 128, dtype=np.uint8)
    img_path = os.path.join(img_dir, img_name)
    cv2.imwrite(img_path, img)

    # Create two dummy instance masks
    mask1 = np.zeros((100, 100), dtype=np.uint16)
    mask1[10:30, 10:30] = 1  # Instance 1

    mask2 = np.zeros((100, 100), dtype=np.uint16)
    mask2[50:70, 50:70] = 1  # Instance 1 in a different mask

    # Name masks with Plate_Site pattern matching the image
    mask_name1 = "A1_s1_instance_mask.png"
    mask_name2 = "A1_s1_mask.png"

    cv2.imwrite(os.path.join(mask_dir, mask_name1), mask1)
    cv2.imwrite(os.path.join(mask_dir, mask_name2), mask2)

    return img_dir, mask_dir

# ------------------------
# Dummy augmentation pipeline
# ------------------------
import albumentations as A

def get_augmentation_pipeline(augmentation_config):
    # Ignore augmentation_config for test — use simple identity transform
    return A.Compose([], additional_targets={'mask': 'mask'})

# ------------------------
# Run the test
# ------------------------
if __name__ == "__main__":
    test_dir = "test_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    img_dir, mask_dir = create_dummy_image_and_masks(test_dir)

    # Patch hf.get_image_mask_pairs_from_folder to use your real function
    import helper_functions as hf  # Replace with your actual helper module

    augmented_data = prepare_augmented_training_data(
        image_folder=img_dir,
        mask_folder=mask_dir,
        augmentation_config={},  # Empty config — identity transform
        num_augments=2,
        save_augmented=True,
        save_dir=os.path.join(test_dir, "augmented"),
        verbose=True
    )

    print("\nTest completed.")
    print(f"Number of augmented samples: {len(augmented_data)}")
    print("Sample augmentation data keys:", augmented_data[0].keys())
