import os
import json
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch
from resources import helper_functions as hf
import resources.json_conversion_tools as jc
from resources import scoring_functions as score

mpl.rcParams['figure.dpi'] = 150   # Makes plots crisper
mpl.rcParams['savefig.dpi'] = 150  # Higher quality saved images
mpl.rcParams['figure.autolayout'] = True  # Automatically adjust layout to prevent cut-off labels


def visualize_masks(image_path, pred_json, classes_to_include=None, model='none'):

    original_image = hf.load_image(image_path)

    # ---- Detect dynamic range ----
    if original_image.dtype == np.uint8:
        max_val = 255
    elif original_image.dtype == np.uint16:
        actual_max = original_image.max()
        if actual_max <= 4095:
            max_val = 4095  # 12-bit
        else:
            max_val = 65535  # Full 16-bit range
    else:
        raise ValueError(f"Unsupported image dtype: {original_image.dtype}")

    print(f"Detected image max value: {max_val}")

    # ---- Prepare image for overlay ----
    if len(original_image.shape) == 2:
        stacked_image = np.stack([original_image]*3, axis=-1)
    else:
        stacked_image = original_image.copy()

    # ---- Load multiclass mask and outline ----
    polygon_outline = jc.load_polygon_mask_for_viz_multiclass(
        json_data=pred_json,
        image_shape=original_image.shape,
        mode='outline',
        classes_to_include=classes_to_include
    )

    polygon_mask = jc.load_polygon_mask_for_viz_multiclass(
        json_data=pred_json,
        image_shape=original_image.shape,
        mode='mask',
        classes_to_include=classes_to_include
    )

    # ---- Normalize image for visualization ----
    stacked_vis = stacked_image.astype(np.float32) / max_val
    stacked_vis = np.clip(stacked_vis, 0, 1)

    overlay_vis = stacked_vis.copy()

    # ---- Build color map dynamically ----
    unique_class_vals = np.unique(polygon_outline)
    unique_class_vals = unique_class_vals[unique_class_vals > 0]  # exclude background

    num_classes = len(unique_class_vals)

    # Use maximally distinct colors (HSV spaced)
    colors = hf.generate_distinct_colors(num_classes)

    class_colors = {}
    for idx, class_val in enumerate(unique_class_vals):
        class_colors[class_val] = colors[idx]

    # ---- Apply class-colored outlines ----
    for class_val, color in class_colors.items():
        outline_indices = np.where(polygon_outline == class_val)
        if outline_indices[0].size > 0:
            overlay_vis[outline_indices[0], outline_indices[1], :] = color

    # ---- Plot ----
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(stacked_vis, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(model.capitalize() + " Outlines (Multiclass Overlay)")
    plt.imshow(overlay_vis)
    plt.axis('off')

    # ---- Legend ----
    legend_elements = []
    for class_val, color in class_colors.items():
        patch = Patch(facecolor=color, edgecolor=color, label=f'Class {class_val - 1}')  # Subtract 1 to match original class keys
        legend_elements.append(patch)

    plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1.05, 0), borderaxespad=0.)

    plt.subplot(1, 3, 3)
    plt.title(model.capitalize() + "  Mask (Filled)")
    plt.imshow(polygon_mask, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_comparison_outlines(image_path, pred_json, gt_json, pred_class=None,
                                  gt_class=None):
    """
    Visualize an image with outlines from two different annotation sources in different colors.
    Also compute and return IoU and Dice scores.
    """

    # ---- Define colors ----
    color_1 = (1.0, 0.0, 0.0)   # Red for inference
    color_2 = (0.0, 1.0, 0.0)   # Green for annotation
    overlap_color = (1.0, 1.0, 0.0)  # Yellow for overlap

    original_image = hf.load_image(image_path)

    # ---- Detect dynamic range ----
    if original_image.dtype == np.uint8:
        max_val = 255
    elif original_image.dtype == np.uint16:
        actual_max = original_image.max()
        if actual_max <= 4095:
            max_val = 4095  # 12-bit
        else:
            max_val = 65535
    else:
        raise ValueError(f"Unsupported image dtype: {original_image.dtype}")

    print(f"Detected image max value: {max_val}")

    # ---- Prepare image for overlay ----
    if len(original_image.shape) == 2:
        stacked_image = np.stack([original_image] * 3, axis=-1)
    else:
        stacked_image = original_image.copy()

    stacked_vis = stacked_image.astype(np.float32) / max_val
    stacked_vis = np.clip(stacked_vis, 0, 1)

    overlay_vis = stacked_vis.copy()

    # ---- Load outlines ----
    polygon_outline_1 = jc.load_polygon_mask_for_viz_multiclass(
        json_data=pred_json,
        image_shape=original_image.shape,
        mode='outline',
        classes_to_include=pred_class
    )

    polygon_outline_2 = jc.load_polygon_mask_for_viz_multiclass(
        json_data=gt_json,
        image_shape=original_image.shape,
        mode='outline',
        classes_to_include=gt_class
    )

    # ---- Load filled masks ----
    mask_1 = jc.load_polygon_mask_for_viz_multiclass(
        json_data=pred_json,
        image_shape=original_image.shape,
        mode='mask',
        classes_to_include=pred_class
    )

    mask_2 = jc.load_polygon_mask_for_viz_multiclass(
        json_data=gt_json,
        image_shape=original_image.shape,
        mode='mask',
        classes_to_include=gt_class
    )

    binary_inference_mask = (mask_1 > 0).astype(np.uint8)

    # ---- Apply outlines ----
    indices_1 = np.where(polygon_outline_1 > 0)
    indices_2 = np.where(polygon_outline_2 > 0)

    # Find overlaps
    overlap_mask = np.logical_and(polygon_outline_1 > 0, polygon_outline_2 > 0)
    overlap_indices = np.where(overlap_mask)

    # Apply colors
    if indices_1[0].size > 0:
        overlay_vis[indices_1[0], indices_1[1], :] = color_1

    if indices_2[0].size > 0:
        overlay_vis[indices_2[0], indices_2[1], :] = color_2

    if overlap_indices[0].size > 0:
        overlay_vis[overlap_indices[0], overlap_indices[1], :] = overlap_color

    # ---- Scoring ----
    print("\n📊 Scoring Results:")

    overall_iou = score.compute_iou(binary_inference_mask, (mask_2 > 0))
    overall_dice = score.compute_dice(binary_inference_mask, (mask_2 > 0))
    print(f"Overall IoU:  {overall_iou:.4f}")
    print(f"Overall Dice: {overall_dice:.4f}")

    # ---- Per-class scores ----
    if gt_class is not None:
        annotation_class_ids = [int(cid) for cid in gt_class]
    else:
        annotation_class_ids = list(np.unique(mask_2))
        annotation_class_ids = [cid - 1 for cid in annotation_class_ids if cid > 0]  # Adjust back

    per_class_scores = score.compute_per_class_scores(binary_inference_mask, mask_2, annotation_class_ids)

    print("\nPer-Class Scores:")
    for cid, scores in per_class_scores.items():
        print(f"Class {cid}: IoU = {scores['IoU']:.4f}, Dice = {scores['Dice']:.4f}")

    # Ensure class IDs are JSON-safe (int keys)
    per_class_scores = {int(k): v for k, v in per_class_scores.items()}

    # ---- Prepare JSON-like result ----
    results = {
        "overall": {
            "IoU": overall_iou,
            "Dice": overall_dice
        },
        "per_class": per_class_scores
    }

    # ---- Plot ----
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_vis)
    plt.axis('off')

    # ---- Improved title ----
    plt.title("Comparison of Inference vs. Annotation\n(Outlines & Overlaps Highlighted)",
              fontsize=14, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_1, edgecolor=color_1, label='Inference'),
        Patch(facecolor=color_2, edgecolor=color_2, label='Annotation'),
        Patch(facecolor=overlap_color, edgecolor=overlap_color, label='Overlap')
    ]
    plt.legend(handles=legend_elements,
               loc='lower left',
               bbox_to_anchor=(1.05, 0),
               borderaxespad=0.)

    # ---- Add score text BELOW the image but inside the plot ----
    score_text = (r"$\bf{Overall\ Scores}$" + "\n"
                  r"$\bf{IoU}$:  " + f"{overall_iou:.4f}    |    " +
                  r"$\bf{Dice}$:  " + f"{overall_dice:.4f}")

    plt.gcf().text(0.5, 0.02,  # Slightly above the bottom edge
                   score_text,
                   fontsize=12,
                   ha='center',
                   va='bottom')

    plt.tight_layout()
    plt.show()

    return results

def batch_visualize_masks(image_inputs, json_inputs, classes_to_include=None, model='none', verbose=False):
    """
    Batch visualization for microscopy images and JSON masks based on Well_Site keys.

    Parameters:
    - image_inputs: Directory path or list of image file paths.
    - json_inputs: Directory path or list of JSON file paths.
    - classes_to_include: Optional list of class IDs to visualize.
    - model: Name of the model (for plot title).
    - verbose: Print matching and processing logs.
    """

    image_files = hf.resolve_files(image_inputs, {'.tif', '.tiff', '.png', '.jpg'})
    json_files = hf.resolve_files(json_inputs, {'.json'})

    # --- Index JSONs by well_site ---
    json_index = defaultdict(list)
    for json_path in json_files:
        key = hf.extract_well_site(Path(json_path).stem)
        if key:
            json_index[key].append(json_path)

    # --- Match image files to JSONs ---
    matched_image_paths = []
    matched_json_paths = []

    for image_path in image_files:
        key = hf.extract_well_site(Path(image_path).stem)
        if not key:
            if verbose:
                print(f"[SKIP] Could not extract key from image: {image_path}")
            continue

        matching_jsons = json_index.get(key)
        if matching_jsons:
            matched_image_paths.append(image_path)
            matched_json_paths.append(matching_jsons[0])  # First match only
            if verbose:
                print(f"[MATCH] {os.path.basename(image_path)} ↔ {os.path.basename(matching_jsons[0])}")
        else:
            if verbose:
                print(f"[NO MASK] No matching JSON for {os.path.basename(image_path)}")

    if not matched_image_paths:
        raise RuntimeError("No matching image/JSON pairs found.")

    # --- Visualize ---
    for idx, (img_path, json_path) in enumerate(zip(matched_image_paths, matched_json_paths), 1):
        with open(json_path, 'r') as f:
            pred_data = json.load(f)

        print(f"\n--- [{idx}/{len(matched_image_paths)}] Visualizing {os.path.basename(img_path)} ---")

        visualize_masks(
            image_path=img_path,
            pred_json=pred_data,
            classes_to_include=classes_to_include,
            model=model
        )

def batch_visualize_and_score(image_json_pairs, pred_class=None,
                              gt_class=None, output_dir="batch_results",
                              save_visualizations=False):
    """
    Run visualize_comparison_outlines in batch mode.

    Args:
        image_json_pairs (list): List of tuples:
            (image_path, pred_json, gt_json)
        pred_class (list or None): List of class keys to include from prediction JSON,
            or None to include all.
        gt_class (list or None): List of class keys to include from ground truth JSON,
            or None to include all.
        output_dir (str): Directory to save visualizations and results.
        save_visualizations (bool): Whether to save comparison visualizations.

    Returns:
        dict: Batch results mapping image names to IoU/Dice scores.
    """
    os.makedirs(output_dir, exist_ok=True)

    batch_results = {}

    for idx, (image_path, pred_json, gt_json) in enumerate(image_json_pairs):
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        print(f"\n🔍 Processing {image_name} ({idx + 1}/{len(image_json_pairs)})")

        # ---- Run visualization and get scores ----
        scores = visualize_comparison_outlines(
            image_path=image_path,
            pred_json=pred_json,
            gt_json=gt_json,
            pred_class=pred_class,
            gt_class=gt_class
        )

        # ---- Store results ----
        batch_results[image_name] = scores

        # ---- Optionally save the figure ----
        if save_visualizations:
            fig_path = os.path.join(output_dir, f"{image_name}_comparison.png")
            plt.savefig(fig_path, bbox_inches='tight')
            print(f"✅ Saved visualization for {image_name}")
            plt.close()  # Prevent memory buildup

    # ---- Save all scores to JSON ----
    score_file = os.path.join(output_dir, "batch_scores.json")
    with open(score_file, "w") as f:
        json.dump(batch_results, f, indent=4)
    print(f"\n✅ All scores saved to {score_file}")

    return batch_results


if __name__ == "__main__":
    image_path = "../data/images/Araceli_A6_s2_w1_z0_1020e47f-73ff-427f-b5aa-44d2915e9068.tiff"
    json_path = "../data/annotations/A6_s2.json"
    json_path_2 = "../data/annotations/A7_s4.json"

    json_data = hf.load_json(json_path)
    json_data_2 = hf.load_json(json_path_2)

    # visualize_masks(image_path, json_data, classes_to_include=['0'])  # None = all classes

    # visualize_comparison_outlines(image_path, json_data, json_data_2, inference_class=['0'], annotation_class=['0'])

    # batch_visualize_masks([image_path], [json_path], model='single_images_test', verbose=True)
    # batch_visualize_masks('../data/images', '../data/annotations', model='directory_test',
    #                       verbose=True)
