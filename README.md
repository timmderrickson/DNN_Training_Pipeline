# 🧠 Deep Learning Training Pipeline

A modular and extensible pipeline for training, evaluating, and visualizing segmentation models on microscopy images. Built around Cellpose, ResUNet (ONNX), and generative augmentation using GANs/VAEs.

---

## 🚀 Features

- 🔄 **Batch Inference** with support for tiling, large image handling, and visual ground truth comparison
- 📐 **Tiled Prediction** for both Cellpose and ONNX-based models
- 🧪 **Flexible Training** pipeline: data prep, augmentation, splitting, training, logging
- 🎨 **Rich Visualizations**: overlay masks, outlines, prediction vs GT analysis
- 📊 **Metric Logging**: IoU, Dice, batch-level exports
- 🧰 **Custom Configs**: model selection, augmentation, checkpointing
- 🧬 **Generative Modules**: supports VAE- and GAN-based synthetic data generation

---

## 📁 Project Structure

```bash
DNN_Training_Pipeline/
├── main.py                      # Entry point for training and batch inference
├── batch_results/               # Output directory for predictions
├── data/
│   ├── images/                  # Raw input TIFF images
│   ├── annotations/            # Ground truth JSONs
│   └── predictions/            # Inference predictions
├── models/
│   ├── trained_models/         # Model checkpoints (ResUNet, Cellpose)
│   └── trained_gan/            # GAN-generated models/data
├── outputs/                    # Visualization results and metric exports
├── resources/                  # Core utilities
├── training/
│   └── training_data/          # Prepared training & validation sets
├── testing/                    # Test datasets and validation splits
├── requirements.gpu.txt        # Package dependencies (GPU-enabled)
└── README.md                   # You are here
```

---

## ⚡ Quickstart

### ▶️ Inference

```python
from main import batch_run
from resources import model_functions as mf

model = mf.instantiate_cellpose_model(net="CPnetV2", gpu=True)

batch_run(
    image_inputs="data/images/",
    ground_truth_json_paths="data/annotations/",
    model_instance=model,
    model_name="cellpose",
    diameter=30,
    compare=True,
    save_visuals=True,
    visualizations=True,
    preprocess_image={"tile": True, "tile_size": (512, 512), "overlap": 32}
)
```

✅ For ResUNet (ONNX):
```python
model = mf.instantiate_resunet_model("models/ResNet50_U-Net.onnx", gpu=True)
```

---

### 🎓 Training

```python
from main import batch_train

batch_train(
    gt_json_folder="data/annotations/",
    image_folder="data/images/",
    output_image_folder="training/training_data/prepared_images/",
    output_mask_folder="training/training_data/prepared_masks/",
    image_shape=(3000, 3000),
    augmentation_config={
        "horizontal_flip": 0.5,
        "brightness_range": (0.9, 1.1)
    },
    num_augments=5,
    split_dir="training/training_data/augmented_data_split/",
    save_path="models/trained_models/",
    n_epochs=100,
    batch_size=8,
    gpu=True
)
```

---

## 🧬 Generative Augmentation

Supports:
- Variational Autoencoders (VAEs) with mask-guided reconstruction
- GANs for synthetic image/mask pair generation
- Custom modules in `resources/generative_utils/` (if applicable)

---

## 📚 Contribution Guide

- Write clear docstrings and comments
- Use consistent naming:  
  `Plate_Site.json ⇄ Araceli_Plate_Site_*.tiff`
- Group helper functions in `resources/`
- Run training/inference through `main.py`
- Validate with both Cellpose and ResUNet pipelines

---

## 🛠 Dependencies

```bash
conda create -n dnnpipe python=3.10
pip install -r requirements.gpu.txt
```

🧪 Includes:
- Cellpose >= 4.0
- OpenCV, Torch, ONNXRuntime
- Albumentations, Pandas, TIFffile, Matplotlib

---

## 📷 Sample Output

> Add screenshots of overlay masks, ground truth comparisons, and metric visualizations here for visual appeal.

---

## 🧾 License

For research use only. Contact for commercial licensing.

---
