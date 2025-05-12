

if __name__ == "__main__":

    augmentation_config = {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.5,
        "random_rotation_90": 0.5,
        "scaling_range": (0.9, 1.1),
        "brightness_range": (0.9, 1.1),
        "contrast_range": (0.9, 1.1),
        "apply_blur": False,
        "translation_pixels": 0,
        "image_shape": (3000, 3000)
    }

    image_paths = ["../training_data/images/Araceli_A6_s2_w1_z0_1020e47f-73ff-427f-b5aa-44d2915e9068.tiff"]
    mask_paths = ["../training_data/masks/A6_s2_instance_mask.png"]

    aug_pipeline = get_augmentation_pipeline(augmentation_config)

    # Create synthetic sample
    synthetic_image, synthetic_mask = create_synthetic_image_and_mask()

    augmented_data = []

    num_augments_per_image = 5
    save_augmented = False
    save_dir = None

    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        if True:
            print(f"\n[{idx + 1}/{len(image_paths)}] Processing {os.path.basename(img_path)}")

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        for aug_idx in range(1, num_augments_per_image + 1):
            augmented = aug_pipeline(image=image, mask=mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']

            if save_augmented:
                img_filename = f"{base_name}_aug_{aug_idx}.png"
                mask_filename = f"{base_name}_aug_{aug_idx}_mask.png"

                img_save_path = os.path.join(save_dir, img_filename)
                mask_save_path = os.path.join(save_dir, mask_filename)

                cv2.imwrite(img_save_path, aug_img)
                cv2.imwrite(mask_save_path, aug_mask)

                print(f"[SAVED] {img_filename} and {mask_filename}")

            # Store for in-memory use
            augmented_data.append({
                "aug_image": aug_img,
                "aug_mask": aug_mask,
                "original_image_path": img_path,
                "original_mask_path": mask_path,
                "augmentation_idx": aug_idx
            })

    print("\n✅ All augmentations complete.")

    aug_images, aug_masks = extract_image_mask_pairs(augmented_data)

    print(f"Extracted {len(aug_images)} augmented images and masks into memory.")

    # -----------------------------------------------------------
    # Step 5 (optional): Visualize one of the augmented results
    # -----------------------------------------------------------
    sample_idx = 0  # Just pick the first augmented sample
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(aug_images[sample_idx], cmap='gray')
    plt.title("Augmented Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(aug_images[sample_idx], cmap='gray')
    plt.imshow(aug_masks[sample_idx], cmap='jet', alpha=0.4)
    plt.title("Augmented Mask Overlay")
    plt.axis('off')

    plt.show()
