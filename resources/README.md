🧩 resources/ Module

This folder contains core utilities for inference, visualization, metric scoring, and JSON conversions.

📂 Files

helper_functions.py – Common utilities (e.g., resolve_files, extract_well_site, loading images/JSON).

model_functions.py – Cellpose model instantiation and inference wrappers.

json_conversion_tools.py – Functions to convert Cellpose masks to polygon JSONs and vice versa.

polygon_json_visualizations.py – Functions to visualize predictions, comparisons, outlines, and overlays.

scoring_functions.py – Metrics (IoU, Dice), batch score aggregation, CSV export.

🛠 Key Usage

All of these are used internally by:

main.py (for batch_run, batch_train)

training_pipeline.py (during mask prep or augmentation)

📚 Notes

Try to keep these stateless, modular, and reusable.

Avoid circular imports between resources/ and training/.
