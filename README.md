# 🔥 Multimodal Gas Leak Detection System

This project is an AI-powered system designed to detect and classify the presence of harmful gases using **sensor readings** and **thermal images**. It leverages a hybrid deep learning model that combines LSTM (for time-sequenced gas sensor data) and CNN (for thermal imagery), deployed in an interactive **Streamlit web app**.

---

## 🚀 Features

- Real-time gas type prediction using sensor and image data
- Predicts: `NoGas`, `Perfume`, `Smoke`, `Mixture`
- Multimodal model: LSTM + CNN
- Confidence-based predictions (e.g., *"Likely Smoke or LPG"*)
- PDF report generation with sensor readings and predictions
- Interactive Streamlit UI with image preview, sensor table, and result visualization

---

## 🧠 Model Architecture

- **Sensor Branch**: LSTM processes time-series sensor data (`MQ2`–`MQ135`)
- **Image Branch**: CNN processes thermal images (64x64)
- **Fusion Layer**: Outputs combined prediction via softmax classification

---

## 📂 Dataset

This project uses a custom dataset of:

- CSV file with 1600 rows
- 8 Gas Sensor Columns (`MQ2`, `MQ3`, `MQ4`, `MQ5`, `MQ6`, `MQ7`, `MQ8`, `MQ135`)
- Label column: `Gas` (`NoGas`, `Smoke`, `Perfume`, `Mixture`)
- Matched thermal images named like `0_Smoke.png`, `1_NoGas.png`, etc.

> 📦 **Download the dataset** here:  
🔗 [Dataset on Google Drive]([https://drive.google.com/your-dataset-link-here](https://drive.google.com/drive/folders/1y0i6PvWeO-_fZ0D1teW1ZdrL8Dz6ETLE?usp=sharing))

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/gas-leak-detection.git
cd gas-leak-detection
pip install -r requirements.txt
