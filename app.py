import os
import streamlit as st
from model import load_forgery_detection_model, detect_image_noise, extract_prnu, predict_image

# Load the model
model = load_forgery_detection_model()

# Create temp directory if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

# Streamlit UI
st.title("Image Forgery Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    image_path = f"temp/{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the image
    st.image(image_path, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Perform prediction
    is_forged, noise_results = predict_image(image_path)

    # Show results
    if is_forged:
        st.error("The image is forged.")
    else:
        st.success("The image is authentic.")
    
    st.write("Noise analysis results:")
    st.json(noise_results)

# Example usage
if __name__ == '__main__':
    # No need for main function in Streamlit app
    pass
