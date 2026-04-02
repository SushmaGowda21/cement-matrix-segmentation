# Cement Matrix Segmentation using YOLOv8

## 📌 Overview

This project focuses on semantic segmentation of cement matrix components from microscopic images using YOLOv8.
The model identifies and classifies different regions such as cracks, cement matrix, aggregates, and air pores.

The pipeline demonstrates a complete workflow from raw image processing → data augmentation → annotation → YOLO training → result evaluation.

---

## 🚀 Features
 - Data augmentation using Albumentations 
 - Mask-based annotation processing 
 - Conversion from COCO/mask to YOLO format 
 - Training using YOLOv8 (Ultralytics)
 - Evaluation with standard metrics
 - Visualization tools 

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Albumentations
- PyTorch


# Folder structure 
```
cement-matrix-segmentation/
├── data/
│   ├── raw_data/            # Original images + masks
│   ├── augmented_data/      # Augmented images and masks (sample only)
│   └── yolo_format_data/    # YOLO format dataset (images + labels)
├── scripts/
│   ├── augmentation.py      # Data augmentation (image + mask)
│   ├── annotation.py        # Mask -> COCO annotation conversion
│   └── coco_to_yolo.py      # COCO -> YOLO conversion
├── notebook/
│   └── Training_yolo_V8_microscopic_concrete.ipynb     # Training and evaluation notebook
├── results/                 # Model outputs and evaluation results
├── models/                  # Trained model weights 
├── yolo.yaml                # Dataset configuration for YOLOv8
├── requirements.txt
└── README.md
```

# pipeline
 
 - Raw Data
	- Input image + segmentation mask
 
 - Data Augmentation
	- Apply transformations (crop, rotate, flip, etc.)
	- Generate additional training samples
 
 - Annotation
	- Convert color masks into structured annotations
 
 - YOLO Conversion
	- Convert annotations into YOLO format (.txt labels)
 
 - Training
	- Train YOLOv8 model using dataset configuration
 
 - Evaluation
	- Analyze performance using confusion matrix and metrics

# How to Run
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Run augmentation
```bash
python scripts/augmentation.py
```

3. Generate annotations
```bash
python scripts/annotation.py
```

4. Convert to YOLO format
```bash
python scripts/coco_to_yolo.py
```

5. Train model

 Run the notebook:
 
 notebook/Training_yolo_V8_microscopic_concrete.ipynb


# Results

Results

The model performance is evaluated using:

Precision–Recall curves
F1-score curves
Confusion matrix
Sample prediction outputs

Example outputs are available in the results/ folder.

📌 Notes
Only sample data is included in this repository for demonstration.
Full dataset was used during training but is not included due to size constraints.
The project demonstrates the end-to-end ML pipeline for image segmentation.


👩‍💻 Author

Sushma Gowda
M.Sc. Digital Engineering, Germany

