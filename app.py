import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from fpdf import FPDF
import io

# Load models and preprocessing tools
@st.cache_resource(show_spinner=False)
def load_resources():
    # Load both models
    model_full = load_model("gas_model.h5")  # sensor + image
    model_sensor_only = load_model("gas_model_sensor_only.h5")  # sensor only
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return model_full, model_sensor_only, scaler, le

model_full, model_sensor_only, scaler, le = load_resources()

# Streamlit page config
st.set_page_config(page_title="🔍 Gas Leak Detection", layout="wide")
st.title("🧪 Multimodal Gas Leak Detection System")
st.markdown("Upload sensor data CSV and optionally thermal images folder path to detect gas types with AI.")

# Global dataset variable
df = None

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "🎯 Filter", "🔍 Predict", "📥 Download"])

with tab1:
    st.header("📤 Upload Inputs")
    uploaded_csv = st.file_uploader("Upload Sensor CSV File", type=["csv"])
    image_folder = st.text_input("Enter path to the thermal image folder (optional)", value="")

    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            st.session_state["df"] = df
            st.session_state["image_folder"] = image_folder.strip()
            st.success("CSV uploaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

with tab2:
    st.header("🎯 Filter Dataset by Gas")
    if "df" in st.session_state:
        df = st.session_state["df"]
        unique_gases = df["Gas"].unique()
        selected_gas = st.selectbox("Select Gas to Filter", options=["All"] + list(unique_gases))
        if selected_gas != "All":
            filtered_df = df[df["Gas"] == selected_gas].reset_index(drop=True)
        else:
            filtered_df = df.copy()
        st.session_state["filtered_df"] = filtered_df
        st.write(f"Filtered dataset rows: {len(filtered_df)}")
        st.dataframe(filtered_df)
    else:
        st.warning("Please upload a CSV file first in the Upload tab.")

with tab3:
    st.header("🔍 Predictions")
    if "filtered_df" in st.session_state:
        df = st.session_state["filtered_df"]
        image_folder = st.session_state.get("image_folder", "").strip()

        sensor_cols = ['MQ2', 'MQ3', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ135']
        if not all(col in df.columns for col in sensor_cols):
            st.error(f"Dataset missing sensor columns: {sensor_cols}")
        else:
            try:
                X_sensor = df[sensor_cols].values
                X_scaled = scaler.transform(X_sensor)
                X_seq = X_scaled.reshape((X_scaled.shape[0], 1, len(sensor_cols)))

                if image_folder and os.path.isdir(image_folder):
                    # Use combined model with images
                    image_size = (64, 64)
                    X_images = []
                    previews = []
                    for image_name in df["Corresponding Image Name"]:
                        path_jpg = os.path.join(image_folder, image_name + ".jpg")
                        path_png = os.path.join(image_folder, image_name + ".png")
                        img_path = path_jpg if os.path.exists(path_jpg) else (path_png if os.path.exists(path_png) else None)
                        if img_path:
                            img = load_img(img_path, target_size=image_size)
                            img_array = img_to_array(img) / 255.0
                            previews.append(img)
                        else:
                            img_array = np.zeros((64, 64, 3))
                            previews.append(Image.new("RGB", image_size, color=(50, 50, 50)))
                        X_images.append(img_array)
                    X_images = np.array(X_images)

                    preds = model_full.predict([X_seq, X_images])
                else:
                    # Use sensor-only model
                    preds = model_sensor_only.predict(X_seq)
                    previews = [None] * len(df)  # no images to show

                pred_labels = le.inverse_transform(np.argmax(preds, axis=1))
                confidences = np.max(preds, axis=1)

                df["Predicted Gas"] = pred_labels
                df["Confidence (%)"] = (confidences * 100).round(2)
                st.session_state["filtered_df"] = df

                st.success("✅ Prediction complete!")

                # Show results with or without images
                for i in range(min(10, len(df))):
                    st.subheader(f"Sample {i+1}")
                    if previews[i]:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(previews[i], width=150, caption=f"{df['Corresponding Image Name'][i]}")
                        with col2:
                            st.write(f"**Actual Gas:** {df['Gas'].iloc[i]}")
                            st.write(f"**Predicted Gas:** {df['Predicted Gas'].iloc[i]} ✅")
                            st.write(f"**Confidence:** {df['Confidence (%)'].iloc[i]}%")
                    else:
                        st.write(f"**Sample {i+1}**")
                        st.write(f"Actual Gas: {df['Gas'].iloc[i]}")
                        st.write(f"Predicted Gas: {df['Predicted Gas'].iloc[i]} ✅")
                        st.write(f"Confidence: {df['Confidence (%)'].iloc[i]}%")

            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.warning("Please upload and filter data first.")

with tab4:
    st.header("📥 Download Results")

    def create_pdf(dataframe):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Gas Leak Detection Report", ln=True, align="C")

        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Total samples: {len(dataframe)}", ln=True)
        pdf.cell(0, 10, "Summary of Predictions:", ln=True)
        summary = dataframe["Predicted Gas"].value_counts()
        for gas, count in summary.items():
            pdf.cell(0, 10, f"{gas}: {count}", ln=True)

        pdf.ln(10)
        pdf.cell(0, 10, "Detailed Results:", ln=True)

        col_width = pdf.w / 5
        pdf.set_font("Arial", "B", 10)
        headers = ["Sample No", "Actual Gas", "Predicted Gas", "Confidence (%)"]
        for header in headers:
            pdf.cell(col_width, 10, header, border=1)
        pdf.ln()

        pdf.set_font("Arial", "", 10)
        for i, row in dataframe.iterrows():
            pdf.cell(col_width, 10, str(i+1), border=1)
            pdf.cell(col_width, 10, str(row["Gas"]), border=1)
            pdf.cell(col_width, 10, str(row["Predicted Gas"]), border=1)
            pdf.cell(col_width, 10, f'{row["Confidence (%)"]}%', border=1)
            pdf.ln()

        pdf_bytes = pdf.output(dest='S').encode('latin1')
        return pdf_bytes

    if "filtered_df" in st.session_state and "Predicted Gas" in st.session_state["filtered_df"].columns:
        df = st.session_state["filtered_df"]
        csv = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button("Download Prediction CSV", csv, "predictions.csv", "text/csv")

        pdf_data = create_pdf(df)
        st.download_button(
            "Download Prediction PDF Report",
            pdf_data,
            file_name="gas_leak_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No predictions to download yet. Please run prediction first.")
