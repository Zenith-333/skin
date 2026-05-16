import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="mobile_net_skin_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define your 22 classes
class_names = [
    "Acne", "Actinic Keratosis", "Atopic Dermatitis", "Basal Cell Carcinoma",
    "Benign Keratosis", "Dermatofibroma", "Eczema", "Impetigo", "Lichen Planus",
    "Melanoma", "Molluscum Contagiosum", "Psoriasis", "Ringworm", "Rosacea",
    "Scabies", "Seborrheic Keratosis", "Skin Cancer", "Squamous Cell Carcinoma",
    "Tinea Versicolor", "Urticaria (Hives)", "Vitiligo", "Warts"
]

st.title("Skin Disease Classification App (Lite)")
st.write("Upload an image and the model will predict the skin disease class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.write(f"### Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
