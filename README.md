# 💓 Real-Time ECG Monitoring & Alert System using Raspberry Pi, ESP32, and AWS
## 📌 Overview
This repository demonstrates the real time deployment of Convolutional Neural Networks (CNN) based Autoencoders to perform anomaly detection on Electrocardiogram (ECG) data and send alerts in case of abnormality detected. Live ECG samples of heart rhythms are differentiated between normal and abnormal based on the reconstruction error. Model is implemented on Raspberry Pi and the ECG signal Aquisition is done by AD8232 integrated with ESP32.

## 📊 Dataset
This dataset consolidates heartbeat signals from the renowned MIT-BIH Arrhythmia Dataset. With a substantial sample size, it serves as a foundation for training advanced neural networks.
- **File Name:** `ptbdb_normal.csv`, `ptbdb_abnormal.csv`
- **Description:** The datasets contain ECG recordings representing heart rhythms. The former encapsulates normal heartbeats, while the latter captures abnormal rhythms, offering a comprehensive view of cardiac activity variations.

## ⚙️ Training the Model:
### 1. Data Pre-processing:
- Load normal and abnormal datasets.
- Drop the target columns to obtain pure data samples.
- Split the normal dataset into training and testing sets.
### 2. Model Training & Evaluation:
- Train the autoencoder model using different loss functions.
- Determine a threshold value for classification based on the 95th percentile of the reconstruction error on training data.
- Evaluate model performance on combined validation data (normal + anomaly).
- Visualize the reconstructed ECG signals for both normal and anomaly samples.
### 3. Best Model Selection:
- Select the best model based on the minimum average validation error.
### 4. Classification & Metrics Calculation:
- Classify reconstruction errors as either normal or anomaly.
- Calculate and display performance metrics like accuracy, precision, recall, F1-score, and display the confusion matrix.

## 📈 Key Findings
* The model achieved an accuracy of 98.66% in classifying ECG rhythms.
* A recall of 100.00% means the model correctly identified all actual positives.
* The F1 Score, a measure of the model's accuracy considering both precision and recall, stands at 99.82%.
These metrics reflect the model's capability in ECG anomaly detection.



An IoT-based real-time health monitoring system that leverages edge computing and cloud services to detect abnormal ECG patterns and instantly notify healthcare providers via SMS. Designed to assist in **remote cardiac monitoring**, particularly for **critical patients** and **isolated areas**.  

---

## 🚀 Project Overview

This project implements an end-to-end pipeline for real-time ECG signal acquisition, anomaly detection, and automatic emergency notification using:

- 🧠 **Machine Learning (Autoencoder)**
- 🌐 **MQTT Protocol**
- 🧾 **AWS IoT Core & SNS**
- 🧠 **Edge Computing with Raspberry Pi**
- 📡 **Wireless data transmission via ESP32**

---

## 🛠️ System Architecture

```mermaid
graph TD
    A[👤 Patient with ECG Sensor (AD8232)] --> B[📶 ESP32 (MQTT Publisher)]
    B --> C[(🌐 Raspberry Pi MQTT Broker)]
    C --> D[🧠 Autoencoder Model]
    D -->|✅ Normal ECG| E1[✅ No Action]
    D -->|❌ Abnormal ECG (High Reconstruction Error)| E[📤 Publish to AWS IoT Core]
    E --> F[📨 SNS (Simple Notification Service)]
    F --> G[📱 SMS Alert to Hospital Number]
