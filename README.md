# Strawberry Ripeness Detection Using YOLO 🍓🔍

Strawberry detection using YOLO. It detects whether the strawberry is **ripe** or **unripe**.

## Overview
This project utilizes the **YOLO (You Only Look Once) object detection model** to identify strawberries in images and classify them based on their ripeness stage. The model is trained on a **custom-labeled dataset** of strawberries to assist in **automated fruit sorting** and **precision agriculture**.

## Features
- 💡 **Real-time strawberry detection**
- 🍓 **Classifies strawberries as ripe or unripe**
- ⏳ **Fast and efficient YOLO-based detection**
- 💪 **Optimizes harvesting and quality control**

## Technologies Used
- **YOLOv8** (Object Detection & Classification)
- **Python** (Model Implementation)
- **OpenCV** (Image Processing)
- **PyTorch** (Deep Learning Framework)
- **Custom-labeled dataset** (Ripeness Classification)

## Dataset
The dataset consists of labeled images with strawberries classified as **ripe** or **unripe**. Annotation was done using **Roboflow**.

## Model Training
The model was trained using **YOLOv8** with the following steps:
1. **Data Preparation:** Images were annotated and split into training, validation, and test sets.
2. **Training:** The YOLO model was trained using **PyTorch** with **transfer learning**.
3. **Evaluation:** Performance was assessed using **mAP (mean Average Precision)**.

## Results
The trained model achieves **high accuracy in detecting and classifying strawberries**, making it useful for **smart farming applications**.

## Applications
- 🌿 **Automated fruit sorting**
- 🚜 **Smart agriculture and harvesting**
- 📝 **Quality control in food industry**

---

🌱 *This project contributes to smart farming by enabling automated ripeness detection, reducing manual labor, and improving crop quality.* 🌱

