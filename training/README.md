

🧪 training/ Module

Contains logic for preparing datasets, applying augmentation, splitting into train/val, and launching training.

📂 Files

training_pipeline.py – Main functions:

prepare_training_data()

prepare_augmented_training_data()

split_augmented_data()

run_training_from_split()

save_metrics_as_csv(), save_training_summary(), etc.

⚙️ Usage

These functions are chained together in batch_train() in main.py.

📌 Best Practices

Input/output folder paths should be created outside these functions (for reproducibility)

Augmentation configs should be passed as dictionaries (randomness handled internally)

Loss metrics (train/val) are saved automatically at save_path/metrics.npy
