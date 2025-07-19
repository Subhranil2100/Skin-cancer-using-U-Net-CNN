# Skin-cancer-using-U-Net-CNN
A lightweight U-Net model designed for skin lesion segmentation using the ISIC 2018 dataset. It follows an encoder–decoder architecture with skip connections, processing 128×128 RGB images to produce binary masks. Trained with binary cross-entropy and Adam optimizer, it’s fast, efficient, and suitable for medical image analysis.

# 🧠 Skin Cancer Detection using U-Net and CNN

This project uses a two-stage deep learning pipeline to **segment** and **classify** skin lesions (benign or malignant) from dermoscopic images. It leverages a **U-Net model** for lesion segmentation and a **CNN classifier** trained on the segmented data. The project is built and evaluated using the **ISIC 2018 dataset**.

---

## 📂 Project Structure

SkinCancerDetection/
│
├── train_model.py # CNN training script
├── evaluate_and_display.py # Model evaluation & prediction
├── train_unet.py # U-Net training script
├── unet_module.py # U-Net architecture & overlay functions
├── segment_dataset.py # Apply U-Net to segment images
│
├── ISIC2018_Task1-2_Training_Input/ # Raw RGB images (from ISIC 2018)
├── ISIC2018_Task1_Training_GroundTruth/ # Ground-truth masks
│
├── train/ # Training data (segmented or original)
│ ├── benign/
│ └── malignant/
│
├── test/ # Test data (segmented or original)
│ ├── benign/
│ └── malignant/
│
├── to_predict/ # Unseen images to classify
├── classified_results/ # Final predictions with overlays
│
├── extracted_features/ # (Optional) For hybrid pipelines
│
├── melanoma_cnn_model.h5 # Trained CNN model
├── unet_model_lightweight.h5 # Trained U-Net model
│
├── training_performance.png # CNN training curves
├── confusion_matrix.png # Confusion matrix plot
├── classification_report.png # Classification report heatmap
├── metrics_summary.txt # Evaluation metrics
└── to_predict_results.csv # CSV of predictions on new images 

---

## 📊 Dataset Used

### 📁 [ISIC 2018 Skin Lesion Analysis Dataset](https://challenge.isic-archive.com/data/)
- **Input Images:** `ISIC2018_Task1-2_Training_Input/`
- **Masks:** `ISIC2018_Task1_Training_GroundTruth/`
- **Classes:** Benign and Malignant
- **Use:**  Train U-Net for segmentation and CNN for classification.
- **Download link** [ISIC2018_Task1-2_Training_Input](https://challenge.isic-archive.com/data/#2018)

---

## 🚀 Model Overview

### 🔹 U-Net (Segmentation)
- Input: RGB images (128×128)
- Output: Binary lesion mask
- Architecture: Encoder–decoder with skip connections
- Loss: Binary Crossentropy
- Optimizer: Adam

### 🔹 CNN (Classification)
- Input: Segmented lesion image (180×180)
- Architecture: 3 Conv blocks + Dense layers + Dropout
- Output: Benign or Malignant
- Loss: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall, F1

---

## 🔧 How It Works

### 1. **Train U-Net**
python train_unet.py
Trains unet_model_lightweight.h5 on image-mask pairs

Saves model and training plot

2. Segment Dataset
python segment_dataset.py
Segments images in train/ and test/ using the U-Net

Saves segmented versions for CNN training

3. Train CNN Classifier
python train_model.py
Trains CNN on segmented dataset (train/, test/)

Saves melanoma_cnn_model.h5 and training curves

4. Evaluate and Predict
python evaluate_and_display.py
Evaluates model on test set

Computes accuracy, confusion matrix, precision, recall, F1

Classifies unseen images from to_predict/

Generates overlaid result images in classified_results/

✅ Example Outputs
🎯 Accuracy: >85%

📉 training_performance.png: Shows model accuracy and loss

📈 confusion_matrix.png & classification_report.png

🖼️ classified_results/: Shows prediction overlays with label & confidence

📌 Sample Overlay Output
Each image in classified_results/ includes:

Original lesion

Segmented mask overlaid in color

Predicted label (Benign/Malignant)

Confidence score

“Skin Cancer” or “No Cancer” tag

🧠 Model Description (U-Net in 350 chars)
A lightweight U-Net model designed for skin lesion segmentation using the ISIC 2018 dataset. It follows an encoder–decoder architecture with skip connections, processing 128×128 RGB images to produce binary masks. Trained with binary cross-entropy and Adam optimizer, it’s fast, efficient, and suitable for medical image analysis.

🛠️ Requirements
pip install tensorflow numpy pandas opencv-python scikit-learn matplotlib seaborn
📌 Notes
All training images are normalized to [0, 1].

Only .jpg images are used (change filter if using .png etc.).

You can extend this pipeline with InceptionV3 + Random Forest for a hybrid model.



