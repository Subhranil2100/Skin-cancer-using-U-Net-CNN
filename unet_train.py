import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from unet_module import build_unet  # Ensure this file exists in the same directory
from sklearn.model_selection import train_test_split

# Function to load and preprocess images and masks
def load_images_and_masks(image_dir, mask_dir, img_size):
    images, masks = [], []
    for img_name in os.listdir(image_dir):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(image_dir, img_name)
        mask_name = img_name.replace(".jpg", "_segmentation.png")
        mask_path = os.path.join(mask_dir, mask_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Skipping {img_name} due to missing data")
            continue

        img = cv2.resize(img, (img_size, img_size)) / 255.0
        mask = cv2.resize(mask, (img_size, img_size)) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Main lightweight training setup
if __name__ == "__main__":
    image_dir = "ISIC2018_Task1-2_Training_Input"
    mask_dir = "ISIC2018_Task1_Training_GroundTruth"
    img_size = 128  # Lightweight resolution

    print("\U0001F4E5 Loading images and masks...")
    X, y = load_images_and_masks(image_dir, mask_dir, img_size)
    print(f"\u2705 Loaded {len(X)} images.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    print("\U0001F9E0 Building lightweight U-Net...")
    model = build_unet((img_size, img_size, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("\U0001F680 Training model (Epochs = 10)...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=10,
                        batch_size=8,
                        verbose=1)

    model.save("unet_model_lightweight.h5")
    print("\u2705 Lightweight U-Net model saved as unet_model_lightweight.h5")
