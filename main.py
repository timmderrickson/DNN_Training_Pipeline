import os
import csv
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
import models.model_functions as mf
import resources.helper_functions as hf
import resources.training_pipeline as tp
import resources.scoring_functions as score
import resources.json_conversion_tools as jc
import resources.polygon_json_visualizations as viz


def run_inference(image_path, model_instance, model_name='cellpose', diameter=None, output_mask_path='output_mask.json',
                  return_result=False, flow_threshold=None, cellprob_threshold=None, preprocess_image=True,
                  visualization=False):
    """
    Runs inference using a specified model on a single image and saves the output mask as JSON.

    Parameters:
    - image_path: path to the input image file.
    - model_instance: CellposeModel (or compatible model).
    - model_name: string identifier (currently supports 'cellpose').
    - model_type: 'cyto' or 'cyto2' (informational only).
    - diameter: estimated object diameter.
    - channels: list of two integers specifying Cellpose channel config.
    - output_mask_path: path to save output JSON.
    - return_result: if True, also returns the result object.
    - flow_threshold: optional float, Cellpose flow filtering.
    - cellprob_threshold: optional float, Cellpose confidence threshold.
    -visualization: optional bool, whether to visualize the resulting image.

    Returns:
    - output_mask_path or (output_mask_path, result) if return_result=True
    """
    output_mask_path = Path(output_mask_path)
    basename = os.path.basename(image_path)

    print(f"[INFO] Starting inference for: {basename}")
    print(f"       → Diameter: {diameter}")
    if flow_threshold is not None:
        print(f"       → Flow threshold: {flow_threshold}")
    if cellprob_threshold is not None:
        print(f"       → Cell probability threshold: {cellprob_threshold}")

    original_image = hf.load_image(image_path)

    if preprocess_image:
        inference_image = hf.preprocess_image_for_inference(original_image, verbose=True)
    else:
        inference_image = original_image

    if model_name.lower() == 'cellpose':
        print(f"[INFO] Running Cellpose inference...")
        mask, flows = mf.run_cellpose_inference(
            model=model_instance,
            image=inference_image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
        print(f"[INFO] Inference complete. Converting mask to JSON...")
        mask_json = jc.convert_cellpose_mask_to_json(mask)
        result = mask_json
    else:
        raise ValueError(f"[ERROR] Unsupported model '{model_name}' in run_inference.")

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_mask_path, 'w') as f:
        json.dump(result, f)

    print(f"[INFO] Saved mask to: {output_mask_path}")

    if visualization:
        viz.visualize_masks(
            image_path=image_path,
            pred_json=result,
            model=model_name
        )

    return (output_mask_path, result) if return_result else output_mask_path

def batch_inference(image_inputs, model_name='cellpose', diameter=None, channels=[0, 0], output_dir='outputs',
                    batch_size=None, gpu=True, model_instance=None, flow_threshold=None, cellprob_threshold=None,
                    preprocess_image=True, visualization=False, verbose=True):
    """
    Runs inference on a list or directory of TIFF images in batches.

    Parameters:
    - image_inputs: directory path or list of .tif/.tiff image paths.
    - model_name: backend model name (default 'cellpose').
    - diameter: estimated object diameter for Cellpose.
    - channels: Cellpose channel config (currently unused).
    - output_dir: where to save JSON predictions.
    - batch_size: how many images to process at a time.
    - gpu: use GPU if available.
    - model_instance: a pre-loaded Cellpose model (optional).
    - flow_threshold: Cellpose flow filtering threshold.
    - cellprob_threshold: Cellpose segmentation threshold.
    - preprocess_image: normalize image before inference.
    - visualization: visualize the prediction output.
    - verbose: print per-image debug info.

    Returns:
    - List of output JSON paths.
    """

    image_paths = hf.resolve_files(image_inputs, valid_exts={'.tif', '.tiff'})

    if not image_paths:
        raise RuntimeError("No valid TIFF images found.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_json_paths = []

    if model_instance is None:
        if model_name == 'cellpose':
            model_instance = mf.instantiate_cellpose_model(net="CPnetV2", gpu=gpu)
        else:
            raise ValueError(f"[ERROR] Model '{model_name}' is not implemented.")

    def process_batch(batch_paths):
        batch_pred_paths = []
        for img_path in tqdm(batch_paths, desc="Processing batch", unit="image"):
            img_basename = Path(img_path).stem
            output_json = output_dir / f"{img_basename}_{model_name}.json"

            if verbose:
                print(f"\n[INFO] Processing: {img_basename}")

            pred_json = run_inference(
                image_path=img_path,
                model_instance=model_instance,
                model_name=model_name,
                diameter=diameter,
                output_mask_path=output_json,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                preprocess_image=preprocess_image,
                visualization=visualization
            )
            batch_pred_paths.append(pred_json)
        return batch_pred_paths

    if not batch_size or batch_size >= len(image_paths):
        pred_json_paths.extend(process_batch(image_paths))
    else:
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            print(f"\n=== Batch {batch_idx + 1}/{total_batches} ===")
            batch = image_paths[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            pred_json_paths.extend(process_batch(batch))

    print(f"\n✅ Batch inference complete. Saved {len(pred_json_paths)} prediction files.")
    return pred_json_paths


def batch_train(gt_json_folder, image_folder, output_image_folder, output_mask_folder, image_shape, augmentation_config,
                num_augments, split_dir, net='CPnetV2', save_path='training/trained_cellpose_model', n_epochs=500,
                learning_rate=0.2, weight_decay=1e-5, batch_size=8, gpu=True, verbose=True):
    """
    Runs full batch training pipeline: data prep → augment → split → train → save metrics.
    """

    print("\n========== BATCH TRAINING START ==========")

    print("Step 1: Preparing training data...")
    tp.prepare_training_data(
        gt_json_folder      = gt_json_folder,
        image_folder        = image_folder,
        output_image_folder = output_image_folder,
        output_mask_folder  = output_mask_folder,
        image_shape         = image_shape,
        verbose             = verbose
    )

    print("\nStep 2: Applying data augmentation...")
    augmented_data = tp.prepare_augmented_training_data(
        image_folder        = output_image_folder,
        mask_folder         = output_mask_folder,
        augmentation_config = augmentation_config,
        num_augments        = num_augments,
        verbose             = verbose
    )

    print("\nStep 3: Splitting data...")
    tp.split_augmented_data(
        augmented_data = augmented_data,
        train_ratio    = 0.8,
        random_seed    = 42,
        verbose        = verbose,
        save_split     = True,
        save_dir       = split_dir
    )

    print("\nStep 4: Running training...")
    tp.run_training_from_split(
        split_dir       = split_dir,
        net             = net,
        save_path       = save_path,
        n_epochs        = n_epochs,
        learning_rate   = learning_rate,
        weight_decay    = weight_decay,
        batch_size      = batch_size,
        gpu             = gpu,
        verbose         = verbose
    )

    print("\nStep 5: Saving metrics and training summary...")
    metrics_path = os.path.join(save_path, "metrics.npy")
    if not os.path.exists(metrics_path):
        print(f"[ERROR] Metrics file not found at {metrics_path}. Training may have failed or been interrupted.")
        return

    try:
        metrics = np.load(metrics_path, allow_pickle=True).item()
    except Exception as e:
        print(f"[ERROR] Failed to load metrics: {e}")
        return

    tp.save_metrics_as_csv(
        metrics_path    = metrics_path,
        output_csv_path = os.path.join(save_path, "metrics.csv")
    )

    tp.save_training_summary(
        n_epochs         = n_epochs,
        learning_rate    = learning_rate,
        weight_decay     = weight_decay,
        batch_size       = batch_size,
        train_loss       = metrics.get('train_loss', []),
        val_loss         = metrics.get('val_loss'),
        save_path        = save_path,
        additional_notes = "Batch training run.",
        pretrained_model = net
    )

    tp.save_final_checkpoint(
        model_save_path = save_path,
        experiment_dir  = "trained_models"
    )

    print("\n✅ Batch training complete!\n")

def batch_run(image_inputs, ground_truth_json_paths=None, prediction_json_paths=None,
              model_instance=None, model_name='cellpose', diameter=None, output_dir='data/predictions',
              batch_size=None, save_visuals=False, pred_class=None, gt_class=None, compare=False,
              overwrite_inference=False, gpu=True, visualizations=True,
              flow_threshold=None, cellprob_threshold=None, preprocess_image=True, verbose=True,
              headless=False, summary_csv_path="batch_results/batch_run_summary.csv"):
    """
    Full batch pipeline: resolves images, runs inference, matches GT if available,
    and performs scoring + visualization (comparison or standalone).

    Accepts either a folder or list for both images and GT JSONs.
    """

    print("\n========== BATCH RUN SETTINGS ==========")
    print(f"Model: {model_name}")
    print(f"Diameter: {diameter}")
    print(f"Compare: {compare}")
    print(f"Prediction classes: {pred_class or 'ALL'}")
    print(f"Ground truth classes: {gt_class or 'ALL'}")
    print(f"Save visuals: {save_visuals}")
    print(f"Overwrite inference: {overwrite_inference}")
    print(f"Visualizations enabled: {visualizations}")
    print(f"Headless mode: {headless}")
    print("=========================================\n")

    image_paths = hf.resolve_files(image_inputs, valid_exts={'.tif', '.tiff'})
    if not image_paths:
        raise RuntimeError("No valid image files found.")

    gt_lookup = {}
    if compare:
        if isinstance(ground_truth_json_paths, str) and os.path.isdir(ground_truth_json_paths):
            gt_files = hf.resolve_files(ground_truth_json_paths, valid_exts={'.json'})
        elif isinstance(ground_truth_json_paths, list):
            gt_files = ground_truth_json_paths
        else:
            raise ValueError("ground_truth_json_paths must be a directory or list of .json paths when compare=True")

        for f in gt_files:
            key = hf.extract_well_site(Path(f).stem)
            if key:
                gt_lookup[key.lower()] = f

    if model_instance is None:
        raise ValueError("You must provide a pre-instantiated model_instance (e.g. from instantiate_cellpose_model()).")

    run_inference_flag = False
    if overwrite_inference:
        print("[INFO] overwrite_inference=True → forcing inference.")
        run_inference_flag = True
    elif prediction_json_paths is None:
        print("[INFO] No prediction JSONs provided → running inference.")
        run_inference_flag = True
    elif len(prediction_json_paths) != len(image_paths):
        raise ValueError("Number of prediction JSONs must match number of images.")
    else:
        print("[INFO] Using provided prediction JSONs.")

    if run_inference_flag:
        prediction_json_paths = batch_inference(
            image_inputs=image_paths,
            model_name=model_name,
            diameter=diameter,
            output_dir=output_dir,
            batch_size=batch_size,
            gpu=gpu,
            model_instance=model_instance,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            preprocess_image=preprocess_image,
            visualization=visualizations and not headless,
            verbose=verbose
        )

    image_json_pairs = []
    summary_rows = []
    for img_path, pred_path in zip(image_paths, prediction_json_paths):
        pred_json = hf.load_json(pred_path)
        gt_json = None

        if compare:
            key = hf.extract_well_site(Path(img_path).stem)
            if key:
                gt_path = gt_lookup.get(key.lower())
                if gt_path and os.path.exists(gt_path):
                    try:
                        gt_json = hf.load_json(gt_path)
                    except Exception as e:
                        print(f"[WARN] Failed to load GT for {Path(img_path).name}: {e}")
                        gt_json = None
                elif verbose:
                    print(f"[WARN] No ground truth JSON found for {Path(img_path).name} → skipping comparison.")

        image_json_pairs.append((img_path, pred_json, gt_json))
        summary_rows.append({
            "image": Path(img_path).name,
            "has_gt": gt_json is not None,
            "used_for_comparison": compare and gt_json is not None,
            "visualized": False  # will update later
        })

    comparison_pairs = [(img, pred, gt) for img, pred, gt in image_json_pairs if gt is not None]

    if compare and comparison_pairs:
        pred_label = f"pred{'-'.join(pred_class)}" if pred_class else "predALL"
        gt_label = f"gt{'-'.join(gt_class)}" if gt_class else "gtALL"
        results_dir = f"batch_results_{pred_label}_{gt_label}" if save_visuals else "batch_results_temp"

        print("[INFO] Running comparison visualization and scoring...")
        batch_results = viz.batch_visualize_and_score(
            image_json_pairs=comparison_pairs,
            pred_class=pred_class,
            gt_class=gt_class,
            output_dir=results_dir,
            save_visualizations=save_visuals and not headless
        )
        score.save_batch_metrics_to_csv(batch_results, output_csv="batch_results/batch_scores.csv")
        for i, (img, pred, gt) in enumerate(image_json_pairs):
            if gt is not None:
                summary_rows[i]["visualized"] = save_visuals and not headless
    elif compare:
        print("[WARN] No ground truth available — skipping comparison scoring.")

    if visualizations and not headless:
        print("[INFO] Visualizing non-comparison images...")
        for i, (img_path, pred_json, gt_json) in enumerate(image_json_pairs):
            if gt_json is None:
                viz.visualize_masks(
                    image_path=img_path,
                    pred_json=pred_json,
                    classes_to_include=pred_class or ['0'],
                    model=model_name
                )
                summary_rows[i]["visualized"] = True

    Path(summary_csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "has_gt", "used_for_comparison", "visualized"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n📄 Summary written to {summary_csv_path}")

def batch_train_main(arguments):
    """
    Wrapper function to run batch training using provided arguments dictionary.
    """
    print("Running batch_train with the following arguments:")
    for k, v in arguments.items():
        print(f"  {k}: {v}")

    batch_train(**arguments)

def batch_run_main(arguments):
    """
    Wrapper function to run batch processing using provided arguments dictionary.
    """
    print("Running batch_run with the following arguments:")
    for k, v in arguments.items():
        print(f"  {k}: {v}")

    batch_run(**arguments)


# ============================== MAIN ==============================

if __name__ == "__main__":

    image_paths = [
        "data/images/Araceli_A1_s1_w1_z0_baf8acb0-02f2-4d95-b194-6eca230e9937.tiff",
        "data/images/Araceli_A6_s2_w1_z0_1020e47f-73ff-427f-b5aa-44d2915e9068.tiff",
        "data/images/Araceli_A7_s4_w1_z0_af3998a3-849c-47fe-9274-382f3879f87c.tiff"
    ]

    ground_truth_json_paths = [
        "data/annotations/A6_s2.json",
        "data/annotations/A7_s4.json"
    ]

    batch_train_arguments = {
        "gt_json_folder": "data/annotations/",
        "image_folder": "data/images/",
        "output_image_folder": "training/training_data/prepared_images/",
        "output_mask_folder": "training/training_data/prepared_masks/",
        "image_shape": (3000, 3000),
        "augmentation_config": {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.5,
            "random_rotation_90": 0.5,
            "scaling_range": (0.9, 1.1),
            "brightness_range": (0.9, 1.1),
            "contrast_range": (0.9, 1.1),
            "apply_blur": False,
            "translation_pixels": 10,
            "image_shape": (3000, 3000)
        },
        "num_augments": 5,
        "split_dir": "training/training_data/augmented_data_split",
        "net": "CPnetV2",
        "save_path": "models/trained_models",
        "n_epochs": 1,
        "learning_rate": 0.2,
        "weight_decay": 1e-5,
        "batch_size": 8,
        "gpu": True,
        "verbose": True
    }

    batch_run_arguments = {
    "image_inputs": "data/images/",  # directory of .tif images
    "ground_truth_json_paths": "data/annotations/",  # directory of .json files
    "model_instance": mf.instantiate_cellpose_model(net="CPnet", gpu=True),
    "model_name": "cellpose",
    "diameter": 30,
    "output_dir": "data/predictions/",
    "batch_size": 2,
    "save_visuals": True,
    "pred_class": ['0', '1'],
    "gt_class": ['0', '1'],
    "compare": True,
    "overwrite_inference": True,
    "gpu": True,
    "visualizations": True,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "preprocess_image": True,
    "verbose": True,
    "headless": False
}

    print("\n===== Running Batch Inference =====\n")
    # batch_run_main(batch_run_arguments)

    print("\n===== Running Batch Training =====\n")
    batch_train_main(batch_train_arguments)
