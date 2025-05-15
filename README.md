# 🧠 Deep Learning Training Pipeline

This project provides a modular and extensible framework for training, evaluating, and visualizing segmentation models for microscopy images using [Cellpose](https://github.com/MouseLand/cellpose) and related tools.

## 🚀 Features

* **Batch inference** on large microscopy datasets with support for ground truth matching, scoring, and visualization.
* **Flexible training pipeline** with automatic data preparation, augmentation, splitting, training, and metrics logging.
* **Rich visualizations**: overlay outlines, masks, comparison plots, and per-class analysis.
* **Metric logging** and batch-level comparison (IoU, Dice, etc.).
* **Custom configuration** for classes, training augmentation, and model checkpointing.

---

## 📦 Directory Structure (Key Parts)

```
DNN_Training_Pipeline/
├── main.py                      # Entry point for training and batch run
├── resources/                   # Utilities and core logic
│   ├── helper_functions.py      # File I/O, parsing, resolution helpers
│   ├── model_functions.py       # Cellpose loading/inference
│   ├── json_conversion_tools.py # JSON ↔ mask conversion
│   ├── polygon_json_visualizations.py  # Overlay, outlines, GT comparison
│   └── scoring_functions.py     # Metric calculations and batch-level exports
│   └── README.md                # Docs for this folder
├── training/
│   ├── training_pipeline.py     # Data prep, augmentation, training logic
│   └── README.md                # Docs for training module
├── models/                      # Saved models and checkpoints
├── data/                        # Input images and annotation JSONs
├── outputs/                     # Predictions, visualizations, and metrics
├── requirements.gpu.txt         # Required packages (GPU-focused)
└── README.md                    # You are here
```

---

## ⚡ Quickstart

### ▶️ Inference

```python
from main import batch_run
from models import model_functions as mf

model = mf.instantiate_cellpose_model(net="CPnetV2", gpu=True)

batch_run(
    image_inputs="data/images/",
    ground_truth_json_paths="data/annotations/",
    model_instance=model,
    diameter=30,
    compare=True,
    save_visuals=True,
    visualizations=True
)
```

### 🎓 Training

```python
from main import batch_train

batch_train(
    gt_json_folder="data/annotations/",
    image_folder="data/images/",
    output_image_folder="training/prepared_images/",
    output_mask_folder="training/prepared_masks/",
    image_shape=(3000, 3000),
    augmentation_config={"horizontal_flip": 0.5, "brightness_range": (0.9, 1.1)},
    num_augments=5,
    split_dir="training/split/",
    save_path="training/trained_model/",
    n_epochs=100,
    batch_size=8,
    gpu=True
)
```

---

## 📚 How To Contribute

* Add clear docstrings to all new functions.
* Group utility functions in `resources/`.
* Follow the naming patterns for ground truth: `Plate_Site.json` ⇄ `Araceli_Plate_Site_*.tiff`
* Run and log training/inference through `main.py` whenever possible.
