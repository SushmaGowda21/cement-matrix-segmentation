import albumentations as A
import cv2
import os
import numpy as np
import re

# Corrected mapping from category names to their corresponding colors in RGB format
category_colors = {
    "cracks": {
        "color": (0, 0, 255),     # Blue
        "id": 1,
    },
    "cement_matrix": {
        "color": (0, 255, 0),     # Green
        "id": 2,
    },
    "gravel_crushed_stones": {
        "color": (255, 0, 0),     # Red
        "id": 3,
    },
    "air_pores": {
        "color": (255, 0, 255),   # Magenta
        "id": 4,
    }
}

# Define the augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),     # Randomly crop the image to 256x256
    A.HorizontalFlip(p=0.5),                 # Horizontally flip the image with a probability of 50%
    A.RandomBrightnessContrast(p=0.2),       # Adjust brightness/contrast with 20% probability
    A.Rotate(limit=40, p=0.5),               # Rotate by a random angle within ±40 degrees
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),  # Shift, scale, and rotate
    A.GaussianBlur(blur_limit=3, p=0.2),     # Apply Gaussian blur with 20% probability
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),  # Adjust color
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=0.2),  # Elastic deformation with alpha_affine set to None
    A.GridDistortion(p=0.2),                 # Apply grid distortion with a 20% probability
    A.Perspective(scale=(0.05, 0.1), p=0.2), # Apply perspective transformation with a 20% chance
], additional_targets={'mask': 'mask'})  # Ensure that the mask is also included in transformations

def get_next_filename(base_filename, extension, num_existing_files):
    # Generate the next file number
    num = num_existing_files + 1
    return f"{base_filename}_transformed_{num}{extension}"

def find_existing_files(base_filename, directory):
    # Regex to match filenames and extract numbers
    pattern = re.compile(rf"{re.escape(base_filename)}_transformed_(\d+)")
    max_num = -1
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num

def save_transformed_images(image, mask, base_filename, num_augmentations, image_save_dir, mask_save_dir_base):
    # Find the highest number used in existing filenames
    num_existing_files = find_existing_files(base_filename, image_save_dir)
    
    for i in range(num_augmentations):
        # Apply the augmentation pipeline
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        
        # Generate the next filename for the augmented image (now saving in .png)
        image_filename = get_next_filename(base_filename, '.png', num_existing_files)
        
        # Save the augmented image as PNG
        cv2.imwrite(os.path.join(image_save_dir, image_filename), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
        
        # Save binary masks for each category using the same base filename
        for category_name, category_info in category_colors.items():
            color_rgb = category_info['color']
            category_id = category_info['id']
            category_save_dir = os.path.join(mask_save_dir_base, category_name)
            os.makedirs(category_save_dir, exist_ok=True)

            # Create a binary mask for the current category
            lower = np.array(color_rgb, dtype=np.uint8)
            upper = np.array(color_rgb, dtype=np.uint8)
            binary_mask = cv2.inRange(transformed_mask, lower, upper)

            # Save the binary mask with the same name as the augmented image
            mask_filename = image_filename  # Use the same filename for the mask (now in .png)
            cv2.imwrite(os.path.join(category_save_dir, mask_filename), binary_mask)

        # Increment the file number
        num_existing_files += 1
        
        print(f"Saved {image_filename} to {image_save_dir} and binary masks to their respective category folders.")

# Paths to your image and mask
image_path = "Prozess_197_Detail_06.tif"
mask_path = "Prozess_197_Detail_06_maske.png"
image_save_dir = "input/train_images"
mask_save_dir_base = "input/train_masks"

# Ensure the save directories exist
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(mask_save_dir_base, exist_ok=True)

# Load the image and color mask in RGB format
image = cv2.imread(image_path)  # OpenCV reads images in BGR format by default
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

mask = cv2.imread(mask_path)  # Load color mask directly in BGR format
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Convert mask from BGR to RGB

# Base filename for saving
base_filename = os.path.splitext(os.path.basename(image_path))[0]

# Number of augmentations you want to generate
num_augmentations = 10

# Save the augmented images and masks
save_transformed_images(image, mask, base_filename, num_augmentations, image_save_dir, mask_save_dir_base)
