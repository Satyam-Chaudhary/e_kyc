import cv2
import numpy as np
import os
import logging
from utils import read_yaml, file_exists

# Logging configuration
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

# Load config
config_path = "config.yaml"
config = read_yaml(config_path)

artifacts = config['artifacts']
intermediate_dir_path = artifacts['INTERMIDEIATE_DIR']
contour_file_name = artifacts['CONTOUR_FILE']


def read_image(image_file, is_uploaded=False):
    """
    Reads an image from a file or Streamlit file uploader.
    """
    try:
        if image_file is None:
            logging.error("read_image called with None image_file.")
            return None

        if is_uploaded:
            # Handle file uploaded via Streamlit
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            # Handle file path
            img = cv2.imread(image_file)

        if img is None:
            logging.error(f"Failed to read image from: {image_file}")
            return None

        logging.info("Image read successfully.")
        return img

    except Exception as e:
        logging.exception(f"Error reading image: {e}")
        return None


def extract_id_card(img):
    """
    Extracts the ID card region from the given image.
    """
    if img is None:
        logging.error("extract_id_card called with None image.")
        return None, None

    try:
        # Convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Noise reduction
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logging.warning("No contours found for ID card.")
            return None, None

        # Select largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        logging.info(f"Largest contour for ID card found at: x={x}, y={y}, w={w}, h={h}")

        # Crop and save ID card image
        contour_id = img[y:y + h, x:x + w]
        current_wd = os.getcwd()
        filename = os.path.join(current_wd, intermediate_dir_path, contour_file_name)

        if file_exists(filename):
            os.remove(filename)
            logging.info(f"Old contour file removed: {filename}")

        cv2.imwrite(filename, contour_id)
        logging.info(f"ID card contour saved at: {filename}")

        return contour_id, filename

    except Exception as e:
        logging.exception(f"Error extracting ID card: {e}")
        return None, None


def save_image(image, filename, path="."):
    """
    Saves an image to the specified path.
    """
    try:
        full_path = os.path.join(path, filename)

        if file_exists(full_path):
            os.remove(full_path)
            logging.info(f"Old file removed: {full_path}")

        cv2.imwrite(full_path, image)
        logging.info(f"Image saved successfully: {full_path}")
        return full_path

    except Exception as e:
        logging.exception(f"Error saving image: {e}")
        return None
