import os
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


def preprocess_image_for_inference(image, verbose=False):
    """
    Preprocesses an input image for inference:
    - Normalizes based on dtype
    - Converts to float32
    - (Optionally extendable for cropping, filtering, etc.)

    Parameters:
    - image: np.ndarray, expected shape (H, W) or (H, W, C)
    - verbose: bool, if True, print debug info

    Returns:
    - norm_image: np.ndarray of float32 scaled to [0, 1]
    """

    if verbose:
        print(f"[PREPROCESS] Original dtype: {image.dtype}, shape: {image.shape}")

    if image.dtype == np.uint8:
        norm_image = image.astype(np.float32) / 255.0

    elif image.dtype == np.uint16:
        max_val = image.max()
        if max_val <= 4095:
            norm_image = image.astype(np.float32) / 4095.0  # 12-bit image
            if verbose:
                print("[PREPROCESS] Detected 12-bit image (max <= 4095)")
        else:
            norm_image = image.astype(np.float32) / 65535.0  # full 16-bit image
            if verbose:
                print("[PREPROCESS] Detected 16-bit image")

    elif image.dtype in (np.float32, np.float64):
        norm_image = np.clip(image, 0, 1).astype(np.float32)
        if verbose:
            print("[PREPROCESS] Image already in float format, clipped to [0, 1]")

    else:
        raise ValueError(f"[PREPROCESS] Unsupported image dtype: {image.dtype}")

    if verbose:
        print(f"[PREPROCESS] Final dtype: {norm_image.dtype}, min: {norm_image.min():.4f}, max: {norm_image.max():.4f}")

    return norm_image

def run_inference(image_path, model_instance, model_name='cellpose', diameter=None, channels=[0, 0],
                  output_mask_path='output_mask.json', return_result=False, flow_threshold=None,
                  cellprob_threshold=None, preprocess_image=True):
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

    Returns:
    - output_mask_path or (output_mask_path, result) if return_result=True
    """
    output_mask_path = Path(output_mask_path)
    basename = os.path.basename(image_path)

    print(f"[INFO] Starting inference for: {basename}")
    print(f"       → Diameter: {diameter}, Channels: {channels}")
    if flow_threshold is not None:
        print(f"       → Flow threshold: {flow_threshold}")
    if cellprob_threshold is not None:
        print(f"       → Cell probability threshold: {cellprob_threshold}")

    original_image = hf.load_image(image_path)

    if preprocess_image:
        inference_image = preprocess_image_for_inference(original_image, verbose=True)
    else:
        inference_image = original_image

    if model_name.lower() == 'cellpose':
        print(f"[INFO] Running Cellpose inference...")
        mask, flows = mf.run_cellpose_inference(
            model=model_instance,
            image=inference_image,
            diameter=diameter,
            channels=channels,
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

    return (output_mask_path, result) if return_result else output_mask_path

def batch_inference(image_paths, model_name, model_type, diameter, channels,
                    output_dir, batch_size=None, gpu=True, model_instance=None):
    """
    Runs inference on images in batches of specified size, with progress bar.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_json_paths = []

    if model_instance is None:
        if model_name == 'cellpose':
            model_instance = mf.instantiate_cellpose_model(model_type=model_type, gpu=gpu)
        else:
            raise ValueError(f"Model '{model_name}' not implemented in batch_inference yet.")

    def process_batch(batch_paths):
        batch_pred_paths = []
        for img_path in tqdm(batch_paths, desc="Processing batch", unit="image"):
            img_basename = Path(img_path).stem
            output_json = output_dir / f"{img_basename}_{model_name}.json"
            pred_json = run_inference(
                image_path=img_path,
                model_instance=model_instance,
                model_name=model_name,
                model_type=model_type,
                diameter=diameter,
                channels=channels,
                output_mask_path=output_json
            )
            batch_pred_paths.append(output_json)
        return batch_pred_paths

    if batch_size is None or batch_size >= len(image_paths):
        pred_json_paths.extend(process_batch(image_paths))
    else:
        for batch_idx in range((len(image_paths) + batch_size - 1) // batch_size):
            print(f"=== Batch {batch_idx + 1} ===")
            batch = image_paths[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            pred_json_paths.extend(process_batch(batch))

    return pred_json_paths

def batch_train(gt_json_folder, image_folder, output_image_folder, output_mask_folder, image_shape, augmentation_config,
                num_augments, split_dir, net='CPnetV2', save_path='training/trained_cellpose_model', n_epochs=500,
                learning_rate=0.2, weight_decay=1e-5, batch_size=8, gpu=True, verbose=True):
    """
    Runs full batch training pipeline: data prep → augment → split → train → save metrics.
    """

    print("\n========== BATCH TRAINING START ==========\n")

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
    metrics = np.load(metrics_path, allow_pickle=True).item()

    tp.save_metrics_as_csv(
        metrics_path    = metrics_path,
        output_csv_path = os.path.join(save_path, "metrics.csv")
    )

    tp.save_training_summary(
        n_epochs         = n_epochs,
        learning_rate    = learning_rate,
        weight_decay     = weight_decay,
        batch_size       = batch_size,
        train_loss       = metrics['train_loss'],
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

def batch_run(image_paths, ground_truth_json_paths=None, prediction_json_paths=None, model_name='cellpose',
              model_type='cyto', diameter=None, channels=[0, 0], output_dir='data/predictions', batch_size=None,
              save_visuals=False, pred_class=None, gt_class=None, compare=False, overwrite_inference=False, gpu=True,
              visualizations=True):
    """
    Runs batch inference and scoring if compare=True, otherwise just visualizes predictions.
    """
    print("\n========== BATCH RUN SETTINGS ==========")
    print(f"Model: {model_name}")
    print(f"Model type: {model_type}")
    print(f"Diameter: {diameter}")
    print(f"Channels: {channels}")
    print(f"Compare: {compare}")
    print(f"Prediction classes: {pred_class or 'ALL'}")
    print(f"Ground truth classes: {gt_class or 'ALL'}")
    print(f"Save visuals: {save_visuals}")
    print(f"Overwrite inference: {overwrite_inference}")
    print(f"Visualizations enabled: {visualizations}")
    print("=======================================\n")

    batch_scores_csv = "batch_results/batch_scores.csv"

    if compare:
        if ground_truth_json_paths is None:
            raise ValueError("ground_truth_json_paths must be provided when compare=True.")
        if len(ground_truth_json_paths) != len(image_paths):
            raise ValueError("Number of ground truth JSONs must match number of images.")

    # 1️⃣ Instantiate the model once
    if model_name == "cellpose":
        model_instance = mf.instantiate_cellpose_model(model_type=model_type, gpu=gpu)
    else:
        raise ValueError(f"Model '{model_name}' is not implemented in batch_run yet.")

    # 2️⃣ Decide if inference is needed
    run_inference_flag = False

    if overwrite_inference:
        print("overwrite_inference=True → running inference regardless of provided predictions.")
        run_inference_flag = True
    elif prediction_json_paths is None:
        print("No prediction JSONs provided. Inference will be run.")
        run_inference_flag = True
    elif len(prediction_json_paths) != len(image_paths):
        raise ValueError("Number of prediction JSONs must match number of images.")
    else:
        print("Using provided prediction JSONs.")

    # 3️⃣ Run inference if needed
    if run_inference_flag:
        prediction_json_paths = batch_inference(
            image_paths=image_paths,
            model_name=model_name,
            model_type=model_type,
            diameter=diameter,
            channels=channels,
            output_dir=output_dir,
            batch_size=batch_size,
            gpu=gpu,
            model_instance=model_instance
        )

    # 4️⃣ Load predictions & GT
    image_json_pairs = []
    for idx, (img_path, pred_path) in enumerate(zip(image_paths, prediction_json_paths)):
        pred_json = hf.load_json(pred_path)
        gt_json = hf.load_json(ground_truth_json_paths[idx]) if ground_truth_json_paths else None
        image_json_pairs.append((img_path, pred_json, gt_json))

    # 5️⃣ If comparing → batch scoring and comparison visualization
    if compare and visualizations:
        if save_visuals:
            pred_label = f"pred{'-'.join(pred_class)}" if pred_class else "predALL"
            gt_label = f"gt{'-'.join(gt_class)}" if gt_class else "gtALL"
            results_dir = f"batch_results_{pred_label}_{gt_label}"
        else:
            results_dir = None

        print("Running batch visualization and scoring...")
        if results_dir is None:
            print("No output directory provided — batch visuals will not be saved.")
        else:
            batch_results = viz.batch_visualize_and_score(
                image_json_pairs=image_json_pairs,
                pred_class=pred_class,
                gt_class=gt_class,
                output_dir=results_dir
            )
            score.save_batch_metrics_to_csv(batch_results, output_csv=batch_scores_csv)

    # 6️⃣ Per-image visualizations
    if visualizations:
        print("Running individual visualizations...")
        for img_path, pred_json, gt_json in tqdm(image_json_pairs, desc="Visualizing images", unit="image"):
            if compare:
                viz.visualize_comparison_outlines(
                    image_path=img_path,
                    pred_json=pred_json,
                    gt_json=gt_json,
                    gt_class=gt_class or ['0']
                )
            else:
                viz.visualize_masks(
                    image_path=img_path,
                    json_data=pred_json,
                    classes_to_include=pred_class or ['0'],
                    model=model_name
                )

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
        "save_path": "training/training/trained_cellpose_model",
        "n_epochs": 1,
        "learning_rate": 0.2,
        "weight_decay": 1e-5,
        "batch_size": 8,
        "gpu": False,
        "verbose": True
    }

    model = mf.instantiate_cellpose_model(net="CPnet", gpu=True)

    out_path = run_inference(
        image_path=image_paths[0],
        model_instance=model,
        model_name='cellpose',
        diameter=30,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )

    print("\n===== Running Batch Inference =====\n")
    # batch_run_main(inference_arguments)

    # print("\n===== Running Batch Training =====\n")
    # batch_train_main(batch_train_arguments)

