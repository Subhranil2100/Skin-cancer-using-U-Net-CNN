import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from unet_module import segment_lesion

# ---------------------- Overlay Function ----------------------
def overlay_segmentation(image, mask, label_text, confidence):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    if len(mask.shape) == 2:
        mask_color = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    else:
        mask_color = mask

    overlay = cv2.addWeighted(image, 0.8, mask_color, 0.4, 0)

    # Compose text
    label = f"{label_text} ({confidence:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_color = (255, 0, 0)      # Blue
    bg_color = (255, 255, 255)    # White

    # Text position and size
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x, y = 10, 30

    # Draw background rectangle
    cv2.rectangle(overlay, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline + 5), bg_color, -1)

    # Put text
    cv2.putText(overlay, label, (x, y), font, font_scale, text_color, thickness)

    return overlay

# ---------------------- Load Models ----------------------
model = load_model("melanoma_cnn_model.h5")
unet_model = load_model("unet_model_lightweight.h5")

# ---------------------- Directories ----------------------
data_dir = "./"
test_dir = os.path.join(data_dir, "test")
to_predict_dir = os.path.join(data_dir, "to_predict")
output_dir = os.path.join(data_dir, "classified_results")
os.makedirs(output_dir, exist_ok=True)

# ---------------------- Evaluate on Test Data ----------------------
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(180, 180),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# ---------------------- Accuracy, Precision, Recall, F1 ----------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall (Sensitivity): {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

# ---------------------- Confusion Matrix ----------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_gen.class_indices, yticklabels=test_gen.class_indices)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# ---------------------- Classification Report ----------------------
report = classification_report(y_true, y_pred, output_dict=True)
plt.figure(figsize=(8, 4))
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='YlGnBu')
plt.title("Classification Report")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "classification_report.png"))
plt.close()

# ---------------------- Sensitivity and Specificity ----------------------
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
else:
    sensitivity = recall
    specificity = None  # Not valid for multiclass

# ---------------------- Save All Metrics ----------------------
with open(os.path.join(output_dir, "metrics_summary.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall (Sensitivity): {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    if specificity is not None:
        f.write(f"Specificity: {specificity:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_true, y_pred))

# ---------------------- Predict on Custom Images ----------------------
print("\n🔍 Classifying images from 'to_predict/'...")
for img_name in os.listdir(to_predict_dir):
    img_path = os.path.join(to_predict_dir, img_name)
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"⚠️ Skipping unreadable image: {img_name}")
        continue

    resized = cv2.resize(orig_img, (180, 180))
    img_array = np.expand_dims(resized / 255.0, axis=0)
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]

    label = "Benign" if class_index == 0 else "Malignant"
    status = "No Cancer" if class_index == 0 else "Skin Cancer"

    # Segment and overlay
    seg_mask = segment_lesion(unet_model, orig_img)
    label_text = f"{label} - {status}"
    segmented_output = overlay_segmentation(orig_img, seg_mask, label_text, confidence * 100)

    # Save result
    save_path = os.path.join(output_dir, f"{label}_{confidence*100:.1f}_{img_name}")
    cv2.imwrite(save_path, segmented_output)

print("✅ Classification completed. Check the 'classified_results/' folder.")
