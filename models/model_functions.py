import os
import numpy as np
import onnxruntime as ort
from cellpose import models
from cellpose.train import train_seg
from resources import helper_functions as hf


def instantiate_cellpose_model(net="CPnetV2", gpu=True, model_path=None):
    """
    Instantiate a Cellpose model (standard or fine-tuned).

    Parameters:
    - net: 'CPnet', 'CPnetV2', or custom (ignored if model_path is given)
    - gpu: use GPU or not
    - model_path: path to a custom-trained model (.pth or .pt)

    Returns:
    - model: a CellposeModel instance ready for inference
    """
    if model_path:
        model = models.CellposeModel(gpu=gpu, pretrained_model=model_path)
        print(f"[INFO] Loaded custom Cellpose model from: {model_path}")
    elif net == "CPnet":
        model = models.CellposeModel(gpu=gpu, pretrained_model="cyto")
        print("[INFO] Loaded default 'cyto' model (CPnet)")
    elif net == "CPnetV2":
        model = models.CellposeModel(gpu=gpu, pretrained_model="cyto2")
        print("[INFO] Loaded default 'cyto2' model (CPnetV2)")
    else:
        raise ValueError(f"Unsupported net type: {net}")

    return model

def run_cellpose_inference(model, image, diameter=None, flow_threshold=None, cellprob_threshold=None):
    """
    Runs Cellpose inference on a single image.

    Parameters:
    - model: A CellposeModel instance
    - image: 2D or 3D numpy array
    - diameter: Estimated object diameter in pixels
    - channels: [cytoplasm, nucleus] channel indices (e.g., [0, 0] for grayscale)
    - flow_threshold: optional float (used to discard bad masks)
    - cellprob_threshold: optional float (threshold for pixel classification)

    Returns:
    - masks: 2D array of instance masks
    - flows: vector flow field (used internally)
    """
    results = model.eval(
        [image],
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    # Handle flexible unpacking
    if len(results) == 4:
        masks, flows, styles, diams = results
    elif len(results) == 3:
        masks, flows, diams = results
        styles = None
    else:
        raise ValueError(f"Unexpected number of return values from model.eval(): {len(results)}")

    return masks[0], flows[0]

def train_cellpose_model(train_images_dir, val_images_dir=None, save_path='trained_cellpose_model', n_epochs=500,
                         learning_rate=0.2, weight_decay=1e-5, batch_size=8, net="CPnetV2", gpu=True):
    """
    Trains a Cellpose model using images and matching _mask files in the same folders.

    Parameters:
        train_images_dir (str): Folder containing training images and _mask files.
        val_images_dir (str): Folder containing validation images and _mask files.
        save_path (str): Where to save the trained model.
        n_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        batch_size (int): Batch size.
        net (str): "CPnet", "CPnetV2", or "OPnet".
            "CPnet"   Classic Cellpose model
            "CPnetV2" Newer default
            "OPnet"   Omnipose model
        gpu (bool): Whether to use GPU.
    """

    # Instantiate Model
    model = instantiate_cellpose_model(net=net, gpu=gpu)

    # Load training data
    print(f"[DEBUG] Loading training data from: {train_images_dir}")
    train_data, train_labels = hf.load_images_and_masks_from_folder(train_images_dir)

    # Load validation data (if provided)
    test_data = None
    test_labels = None

    if val_images_dir:
        print(f"[DEBUG] Loading validation data from: {val_images_dir}")
        test_data, test_labels = hf.load_images_and_masks_from_folder(val_images_dir)

    # Train
    train_seg(
        net=model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        save_path=save_path,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size
    )

def instantiate_resunet_model(model_path, gpu=True):
    """
    Loads the ResNet50-U-Net ONNX model using ONNX Runtime.

    Args:
        model_path (str): Path to the .onnx model file.
        gpu (bool): If True, attempts to use CUDA; else uses CPU.

    Returns:
        ort.InferenceSession: ONNX model session.
    """
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"[INFO] Loaded ResNet50-U-Net ONNX model from {model_path}")
    return session

def run_resunet_inference(model_session, image, input_key='input', output_key='conv2d_8'):
    """
    Runs inference using the ResNet50-U-Net ONNX model on a preprocessed image.

    Args:
        model_session (onnxruntime.InferenceSession): The loaded ONNX model.
        image (np.ndarray): Preprocessed image (H, W) or (H, W, C), float32 [0, 1].
        input_key (str): Input tensor name.
        output_key (str): Output tensor name.

    Returns:
        np.ndarray: Segmentation mask (H, W), binarized.
    """
    import cv2

    # Ensure image is 3-channel RGB (H, W, 3)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # Resize to model input shape
    image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    input_tensor = np.expand_dims(image_resized, axis=0)  # (1, 512, 512, 3)

    # Run inference
    outputs = model_session.run([output_key], {input_key: input_tensor})[0]
    mask = outputs.squeeze()  # (512, 512)

    # Resize back to original image size
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    return (mask_resized > 0.5).astype(np.uint8)

def run_tiled_inference(model_name, model_instance, image, tile_size=(512, 512), overlap=0, normalize=True, **kwargs):
    """
    Runs tiled inference on a large image using the specified model.

    Args:
        model_name (str): Either 'cellpose' or 'resunet'.
        model_instance: Pre-instantiated model object.
        image (np.ndarray): Full-sized image.
        tile_size (tuple): Size of each tile (H, W).
        overlap (int): Number of pixels to overlap between tiles.
        **kwargs: Additional model-specific inference args (e.g., diameter).

    Returns:
        np.ndarray: Reconstructed full-size mask.
    """
    tiles, coords, padded_shape = hf.tile_image(image, tile_size=tile_size, overlap=overlap)
    if normalize:
        tiles = [hf.normalize_image(tile) for tile in tiles]

    tile_preds = []
    for tile in tiles:
        if model_name.lower() == 'cellpose':
            mask, _ = run_cellpose_inference(model_instance, tile, **kwargs)
        elif model_name.lower() == 'resunet':
            mask = run_resunet_inference(model_instance, tile)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        tile_preds.append(mask)

    full_mask = hf.stitch_tiles(tile_preds, coords, padded_shape, tile_size=tile_size)
    return full_mask[:image.shape[0], :image.shape[1]]

# ============================== MAIN ==============================

if __name__ == "__main__":

    # # ✅ Use your actual training folder (already containing images + _mask files)
    # train_images_dir = "../training/training_data/augmented_data_split/train/images"
    # val_images_dir = "../training/training_data/augmented_data_split/val/images"
    #
    # # ✅ Normalize paths for Windows
    # # train_images_dir = os.path.abspath(train_images_dir).replace("\\", "/")
    # # val_images_dir = os.path.abspath(val_images_dir).replace("\\", "/")
    #
    # # ✅ Confirm the files are seen before training
    # print("[TEST] Files in training folder:")
    # print(os.listdir(train_images_dir))
    #
    # # ✅ Run training function
    # train_cellpose_model(
    #     net="CPnetV2",
    #     train_images_dir=train_images_dir,
    #     val_images_dir=val_images_dir,
    #     save_path='training/test_model',
    #     n_epochs=1,  # Short test training
    #     learning_rate=0.2,
    #     weight_decay=1e-5,
    #     batch_size=2,  # Small batch for testing
    #     gpu=True  # Use CPU to avoid GPU errors in test
    # )
    image = "../data/images/Araceli_A6_s2_w1_z0_1020e47f-73ff-427f-b5aa-44d2915e9068.tiff"

    # model = instantiate_cellpose_model(net="CPnetV2", model_path=None, gpu=True)
    #
    image = hf.load_image(image)

    image = hf.preprocess_image_for_inference(image)

    # masks, flows = run_cellpose_inference(model, image, cellprob_threshold=0.5, diameter=10, flow_threshold=0.5)
    resnet = instantiate_resunet_model("ResNet50_U-Net.onnx", gpu=True)

    mask = run_resunet_inference(resnet, image)
    print('done')
