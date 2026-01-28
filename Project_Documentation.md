# Unit 5: Deep Learning and Neural Networks - Project Report

## Introduction
This project implements a sophisticated multi-model object detection system for agricultural quality control. Utilizing the YOLOv11 architecture, the system is designed to identify defects in potatoes, carrots, and lemons, segregating them into specific processing streams to ensure high-quality output and safety.

---

## LO1: Theoretical Underpinning of Neural Networks

### Convolutional Neural Networks (CNNs) vs. Traditional ML
Traditional machine learning (e.g., SVMs) relies on manual feature extraction (HOG, SIFT), which is brittle in varying lighting or orientations. In contrast, **Convolutional Neural Networks (CNNs)** automate feature extraction through heirarchical layers:
1.  **Convolutional Layers**: Apply filters (kernels) to detect edges, textures, and shapes.
2.  **Pooling Layers**: Downsample feature maps to reduce dimensionality and enforce translation invariance.
3.  **Fully Connected Layers**: Aggregate features for final classification.

For this project, **YOLOv11** (You Only Look Once) was selected. It is a one-stage detector that predicts bounding boxes and class probabilities directly from full images in a single forward pass, making it ideal for the real-time constraints of a conveyor belt system.

---

## LO2: Data Engineering & Model Optimization

### Strategy: "Expert Systems"
To address the significant dataset imbalance (Potato: 12k vs. Lemon: 600), three separate "Expert Models" were trained rather than a single monolithic model.

| Dataset | Strategy | Rationale |
| :--- | :--- | :--- |
| **Potato (12k)** | YOLOv11s (Small) | Large dataset supports a deeper network for higher accuracy. |
| **Carrot (Small)** | YOLOv11n (Nano) + Rotation | Cylindrical symmetry allows for aggressive 180° rotation augmentation. |
| **Lemon (600)** | YOLOv11n + HSV Augmentation | The small dataset required **Oversampling** and heavy **Hue/Saturation** augmentation to generalize mold detection across lighting conditions. |

### Augmentation Pipeline
- **Mosaic**: Combines 4 training images to simulate crowded conveyor belts.
- **MixUp**: Blends images to prevent overfitting on the small Lemon dataset.
- **HSV-Hue**: Shifted by 0.015 to simulate varying stages of ripeness and mold colors (blue/green).

---

## LO3: Testing and Evaluation

### Performance Metrics
The models are evaluated using **mAP@0.5** (Mean Average Precision at 50% IoU threshold).
- **Confusion Matrix**: Used to analyze specific misclassifications (e.g., confusing "Unripe Lemon" with "Healthy Lemon").
- **Precision-Recall Curve**: Balances the cost of false positives (waste) vs. false negatives (shipping bad produce).

*(Note: Validation results will be populated in the `runs/detect/` directories after training.)*

---

## LO4: Professional Application & Deployment

### Streamlit Procurement Platform
 The user interface (`app.py`) simulates a factory control station with three distinct tabs.
- **Resource Efficiency**: Models are loaded lazily (only when the specific tab is active) to conserve GPU memory.
- **Video Inference**: Uses `cv2` and `YOLO.track()` to maintain object IDs across frames, allowing for accurate counting of passing items.

### AI Safety Advisor (LLM Integration)
Integrated **Google Gemini GenAI** to provide context-aware safety advice.
- **Trigger**: Defect detection (e.g., "5 detections of Fungal Potato").
- **Output**: Real-time advice on handling specific mycotoxins (e.g., *Solanine* in green potatoes or *Aflatoxins* in moldy crops), ensuring compliance with health and safety regulations.

### Ethical & Societal Impact (C.D3)
**Technologically Mediated Unemployment**: Automated sorting replaces repetitive manual labor. While this increases efficiency and removes humans from hazardous environments (mold spore exposure), it raises ethical questions regarding job displacement in the agricultural sector. The project proposes reskilling workers for system monitoring and maintenance roles.

---

## Distinction Criteria: Explainability (C.P5)
To ensure the "Black Box" of the neural network is interpretable, **Grad-CAM** (Gradient-weighted Class Activation Mapping) is proposed to visualize specific regions of interest. For example, verifying that the Lemon model is activating on the *texture of the mold* rather than the *shape* of the fruit.
