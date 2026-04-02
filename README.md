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
 ```bash
 notebook/Training_yolo_V8_microscopic_concrete.ipynb
```

## 📈 Results & Evaluation

The YOLOv8 segmentation model was evaluated on microscopic concrete images using multiple performance metrics and visual analysis.

---

### 📊 Quantitative Results

The model achieved strong performance across most classes:

- **Cement Matrix:** ~96% accuracy (highest performing class)
- **Gravel / Crushed Stones:** ~89% accuracy
- **Cracks:** ~81% accuracy
- **Air Pores:** ~100% accuracy (limited samples)

---

### 📉 Key Observations

- The model performs very well on structured regions such as cement matrix and aggregates.
- Cracks are slightly more challenging due to thin and irregular shapes.
- Some confusion exists between object classes and background:
  - Objects occasionally predicted as background (false negatives)
  - Background sometimes misclassified as objects (false positives)

---

### 📊 Evaluation Metrics

The model was evaluated using:

- **IoU (Intersection over Union)**
- **mIoU (mean IoU across classes)**
- **Precision**
- **Recall**
- **F1-score**
- **mAP (mean Average Precision)**

#### IoU Formula

IoU = Area of Overlap / Area of Union

This metric measures how well the predicted segmentation matches the ground truth.

---

### 📈 Training Performance

- Training and validation losses decreased steadily, indicating stable learning.
- Precision and recall improved consistently over epochs.
- mAP50 and mAP50–95 scores showed strong convergence.

---

### 🖼️ Visual Results

The model successfully segments:

- Cracks (thin structures)
- Cement matrix regions
- Aggregates (gravel/crushed stones)
- Air pores

Example predictions are available in the `results/` folder.

---

### ⚠️ Limitations

- Some false positives in background regions
- Slight misclassification between visually similar textures
- Performance depends on image quality and annotation accuracy

---

### ✅ Conclusion

The model demonstrates strong capability in segmenting cement matrix components and provides a reliable foundation for automated material analysis in microscopic images.


👩‍💻 Author

Sushma Gowda
M.Sc. Digital Engineering, Germany

