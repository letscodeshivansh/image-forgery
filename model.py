import os
import numpy as np
import tensorflow as tf
from skimage import exposure, img_as_float
from scipy.fft import fft2
import cv2
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
load_dotenv()

# Load the pre-trained model
model = load_model('forgery_detection_model.keras')

img_width, img_height = 224, 224

def load_forgery_detection_model(model_path='forgery_detection_model.keras'):
    """
    Load the pre-trained model.
    Args:
        model_path (str): Path to the model file.
    Returns:
        model: Loaded TensorFlow model.
    """
    model = load_model(model_path)
    return model

# Noise Detection Functions
def detect_image_noise(image_path):
    """
    Detect various types of noise in the image, including JPEG artifacts, lighting inconsistencies, 
    edge artifacts, and other common noise types.
    Args:
        image_path (str): Path to the input image.
    Returns:
        dict: A dictionary of noise metrics and detection flags.
    """
    results = {}
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not open or find the image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_float = img_as_float(gray)

        # 1. Visual Inspection (Histogram)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        if np.std(hist) > 10:
            results['visual_inspection'] = True
        else:
            results['visual_inspection'] = False

        # 2. Statistical Analysis (Variance and Standard Deviation)
        variance = np.var(gray)
        stddev = np.std(gray)
        results['variance'] = variance
        results['stddev'] = stddev

        # 3. Edge Detection (Sobel and Canny)
        edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        edges_canny = cv2.Canny(gray, 100, 200)
        results['sobel_edges'] = np.mean(edges_sobel)
        results['canny_edges'] = np.mean(edges_canny)

        # 4. Signal-to-Noise Ratio (SNR)
        snr = np.mean(image_float) / (np.std(image_float) + 1e-10)
        results['snr'] = snr

        # 5. Frequency Domain (FFT)
        f_transform = fft2(gray)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
        results['fft_noise_level'] = np.mean(magnitude_spectrum)

        # 6. JPEG Artifacts Detection (DCT)
        dct_transformed = cv2.dct(np.float32(gray) / 255.0)
        dct_threshold = np.mean(dct_transformed)
        results['jpeg_artifacts'] = dct_threshold > 0.5

    except Exception as e:
        results['error'] = f"Error processing image: {str(e)}"
    
    return results

def extract_prnu(image_path):
    """
    Extracts PRNU (Photo-Response Non-Uniformity) from the input image.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        numpy.ndarray: PRNU pattern.
    """
    try:
        # Load and convert image to grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not open or find the image: {image_path}")

        # Normalize the image
        image_float = img_as_float(image)

        # Get noise by subtracting the mean
        noise = image_float - np.mean(image_float)

        # Calculate the PRNU as the noise pattern
        prnu = noise / np.linalg.norm(noise)

        return prnu
    except Exception as e:
        print(f"Error extracting PRNU: {str(e)}")
        return None

# Preprocess Image
def preprocess_image(image_path):
    """Preprocess the image for prediction and perform noise detection."""
    noise_results = detect_image_noise(image_path)
    print("Noise Detection Results:", noise_results)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array, noise_results

# Predict if the image is forged or not
def predict_image(image_path, threshold=0.5):
    """Predict if the image is forged or authentic."""
    img_array, noise_results = preprocess_image(image_path)

    # Extract PRNU
    prnu = extract_prnu(image_path)
    noise_results['prnu'] = prnu if prnu is not None else 'Error extracting PRNU'

    # Make the prediction
    prediction = model.predict(img_array)

    # Determine class based on the adjusted threshold
    is_forged = prediction[0][0] > threshold

    return is_forged, noise_results

def predict_batch(image_paths, threshold=0.5):
    """Predict a batch of images without threading."""
    results = []
    for img_path in image_paths:
        image_path, is_forged, noise_results = predict_image(img_path, threshold)
        results.append((image_path, is_forged, noise_results))
    
    # Output results
    for img_path, is_forged, noise_results in results:
        if is_forged:
            print(f'The image {img_path} is forged.')
        else:
            print(f'The image {img_path} is authentic.')
        print(f'Noise analysis for {img_path}:', noise_results)

# # Example usage
# if __name__ == '__main__':
#     test_image_path = 'images/forged3.png'  # Replace with your test image path
#     image_path, is_forged, noise_results = predict_image(test_image_path)
    
#     if is_forged:
#         print(f'The image {image_path} is forged.')
#     else:
#         print(f'The image {image_path} is authentic.')
    
#     print(f'Noise analysis for {image_path}:', noise_results)
