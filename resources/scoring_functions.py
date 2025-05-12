import csv
import numpy as np
import pandas as pd
from pathlib import Path


def compute_iou(mask1, mask2):
    """Compute overall IoU between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0
    return iou

def compute_dice(mask1, mask2):
    """Compute overall Dice coefficient between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    dice = (2. * intersection) / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0
    return dice

def compute_per_class_scores(cellpose_mask, polygon_mask, class_ids):
    """
    Compute IoU and Dice per class between the Cellpose mask and the annotation mask.

    Parameters:
        cellpose_mask (np.ndarray): Binary mask from Cellpose.
        polygon_mask (np.ndarray): Multiclass mask from JSON.
        class_ids (list): List of class IDs (as integers) to compute scores for.

    Returns:
        dict: Per-class IoU and Dice scores.
    """
    results = {}

    for class_id in class_ids:
        # Class value in mask = class_id + 1 (since we added +1 when creating the mask)
        polygon_class_mask = (polygon_mask == (class_id + 1)).astype(np.uint8)

        # Compute IoU and Dice between Cellpose mask and current class
        iou = compute_iou(cellpose_mask, polygon_class_mask)
        dice = compute_dice(cellpose_mask, polygon_class_mask)

        results[int(class_id)] = {'IoU': iou, 'Dice': dice}

    return results

def save_batch_metrics_to_csv(batch_results, output_csv="batch_scores.csv"):
    """
    Saves batch evaluation metrics to a CSV and computes aggregate stats.

    Args:
        batch_results (dict): Mapping from image names to score dicts.
        output_csv (str): Path to output CSV file.
    """

    if not batch_results:
        print("❌ No batch results to save.")
        return

    # Get metric names from first item
    sample_scores = next(iter(batch_results.values()))
    metric_names = list(sample_scores.keys())

    rows = []
    for image_name, scores in batch_results.items():
        row = {"image": image_name}
        row.update(scores)
        rows.append(row)

    # Write to CSV
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image"] + metric_names)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Batch metrics saved to {csv_path}")

    # ---- Aggregate stats ----
    print("\n📊 Aggregate Metrics:")
    for metric in metric_names:
        values = [r[metric] for r in rows if r[metric] is not None]
        if values:
            print(f"  {metric} — Mean: {np.mean(values):.4f} | Median: {np.median(values):.4f}")
        else:
            print(f"  {metric} — No valid scores.")

def log_failure_cases(batch_results, output_path="batch_results/failure_cases.csv",
                      metric="Dice", threshold=0.5, max_report=10):
    """
    Logs images with poor performance (metric below threshold).

    Args:
        batch_results (dict): Mapping from image names to score dicts.
        output_path (str): Where to save the failure cases CSV.
        metric (str): Which metric to use (e.g., 'Dice').
        threshold (float): Failures are metric scores < threshold.
        max_report (int): How many worst cases to print to console.
    """

    # Convert to DataFrame for easy filtering/sorting
    df = pd.DataFrame([
        {"image": img, **scores} for img, scores in batch_results.items()
    ])

    if metric not in df.columns:
        print(f"❌ Metric '{metric}' not found in batch results.")
        return

    # Find failures
    failures = df[df[metric] < threshold].sort_values(by=metric)

    if failures.empty:
        print(f"✅ No failure cases found (all {metric} >= {threshold}).")
        return

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    failures.to_csv(output_file, index=False)
    print(f"⚠ Found {len(failures)} failure cases. Saved to {output_file}.")

    # Print worst N cases
    print(f"\n🔎 Worst {min(max_report, len(failures))} cases:")
    for _, row in failures.head(max_report).iterrows():
        print(f"  {row['image']} — {metric}: {row[metric]:.4f}")
