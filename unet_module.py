# unet_module.py
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np

def build_unet(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)

    # Upsampling
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return models.Model(inputs, outputs, name="U-Net")

def segment_lesion(model, image):
    img_resized = tf.image.resize(image, (128, 128)) / 255.0
    pred_mask = model.predict(tf.expand_dims(img_resized, axis=0))[0]
    pred_mask = (pred_mask > 0.5).astype("uint8")
    return tf.image.resize(pred_mask, (image.shape[0], image.shape[1])).numpy().astype("uint8")

def overlay_segmentation(image, mask, label, confidence):
    overlay = image.copy()
    h, w, _ = image.shape

    # Resize mask to match image
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Ensure mask is binary
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Create a color mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = (0, 255, 0)  # Green overlay

    # Blend mask on image
    blended = cv2.addWeighted(overlay, 0.8, colored_mask, 0.4, 0)

    # Construct full label text
    full_text = f"{label} ({confidence:.1f}%)"

    # Draw background rectangle for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)

    rect_x1, rect_y1 = 5, 5
    rect_x2 = rect_x1 + text_width + 10
    rect_y2 = rect_y1 + text_height + 10

    cv2.rectangle(blended, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
    cv2.putText(blended, full_text, (rect_x1 + 5, rect_y2 - 5), font, font_scale, (0, 0, 255), thickness)

    return blended
