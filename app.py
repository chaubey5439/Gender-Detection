import streamlit as st
import numpy as np
import cv2
import joblib

# Page config
st.set_page_config(page_title="Gender Detector", page_icon="🧠", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 10px;
        }
        .stFileUploader, .stImage, .stTextInput, .stMarkdown {
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Load model, PCA, and face detector
model = joblib.load('gender.pkl')
pca = joblib.load('pca_transform.pkl')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712036.png", width=150)
    st.markdown("## Gender Detection")
    st.markdown("Upload a face image and our trained ML model will predict the gender.")
    st.markdown("---")
    st.info("Model: PCA + Logistic Regression")
    st.success("Accuracy: ~88%")

# Main area
st.title("🎯 Real-time Gender Detection")
st.write("This app detects the **gender of a person from an image** using a Machine Learning model trained on facial features.")

uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_gray, 1.1, 5)

    if len(faces) == 0:
        st.warning("No face detected in the image.")
    else:
        for (x, y, w, h) in faces:
            face = img_gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100)) / 255.0
            face_pca = pca.transform([face.flatten()])
            gender = model.predict(face_pca)[0]

            color = (0, 255, 0) if gender == "male" else (255, 0, 255)
            label_color = "#00cc44" if gender == "male" else "#e600e6"

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, gender.upper(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        st.markdown(f"<h3 style='color:{label_color};text-align:center;'>Predicted Gender: {gender.upper()}</h3>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Face", channels="RGB")

st.markdown("---")
st.markdown("#### ⚙️ Model Details")
with st.expander("See model training info"):
    st.code("""
- Dataset: Labeled face images (Male/Female)
- Image Size: 100x100
- Grayscale Normalization
- PCA for Dimensionality Reduction
- Classifier: Logistic Regression
- Accuracy: ~88%
    """)
