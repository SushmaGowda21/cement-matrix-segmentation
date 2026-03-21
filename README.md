# Cement Matrix Segmentation using YOLOv8

This project focuses on segmenting cement matrix components from microscopic images using YOLOv8. The model identifies and classifies different regions within the cement matrix to support material analysis.

---

## 🚀 Features

- Object detection and segmentation using YOLOv8
- Data preprocessing and augmentation
- Custom dataset training (microscopic concrete images)
- COCO to YOLO format conversion
- Visualization of predictions

---
## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Google Colab

---

## Project Structure
Cement-matrix-segmentation/ │ ├── ColabNotebooks/
│ ├── data/ │ │ ├── Test_images/
│ │ ├── YOLOV8-data/ │ │ │ ├── train/
│ │ │ ├── valid/
│ │ │ └── results/
│ │ └── YOLOv8_object_properties.csv
│ └── Training_yolo_V8_microscopic_concrete.ipynb
│ ├── input/
│ ├── train/
│ ├── val/
│ └── test/
│ ├── augmentation.py
├── coco_to_Yolo.py
├── annotation.py
├── Prozess_197_Detail_06.tif
├── Prozess_197_Detail_06_maske.png
├── yolo.yaml
└── README.md


---

## ⚙️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cement-matrix-segmentation.git



### Step 1: Copy the Colab Notebook and Upload Data

1. **Copy the notebook**:
   - Download the notebook file (`Training_yolo_V8_microscopic_concrete.ipynb`) from this repository.
   - Open it in [Google Colab](https://colab.research.google.com/).

2. **Adjust the `yaml` file**:
   - Modify the `yolo.yaml` file to reflect the correct paths for your dataset:
     ```yaml
     train: data/YOLOV8-data/train
     val:  data/YOLOV8-data/val
     test:data
     nc: 1  # number of classes (e.g., if only cement matrix, set to 1)
     names: ['cement_matrix']  # name of your class
     ```

### Step 2: Perform Data Augmentation and Split Dataset

If you want to augment the dataset and split it into training and testing sets, follow the instructions below:

1. **Run the augmentation script**:
   - Use the provided `augmentation.py` script to automatically augment the data. This script will augment the images and save them in the `input/` folder.
   - Then split the data according to an 80-20 rule (80% for training, 20% for validation).

2. **Annotate your data**:

  - Use the annotation.py script to generate annotations for your dataset. Ensure that all annotations are in COCO format if needed.
  - Convert annotations from COCO to YOLO format:

3. **coco_to_Yolo.py**:
  -Run the coco_to_Yolo.py script to convert COCO annotations into YOLOv8 format**:
  - This script will save the converted annotations in the appropriate folder, ready for YOLOv8 training.


