from deepface import DeepFace
import cv2
import os
import logging
from utils import file_exists, read_yaml

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format=logging_str,
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "ekyc_logs.log"), mode="a"),
        logging.StreamHandler()
    ]
)

config_path = "config.yaml"
config = read_yaml(config_path)
artifacts = config['artifacts']

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
output_path = os.path.normpath(artifacts['INTERMIDEIATE_DIR'])


def detect_and_extract_face(img):
    logging.info("Extracting face...")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        logging.error(f"Failed to load Haar cascade from path: {cascade_path}")
        return None

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    logging.info(f"Detected {len(faces)} faces.")

    if len(faces) == 0:
        logging.warning("No face detected in the image.")
        return None

    largest_face = max(faces, key=lambda f: f[2] * f[3])  # Pick largest
    (x, y, w, h) = largest_face

    new_w, new_h = int(w * 1.5), int(h * 1.5)
    new_x, new_y = max(0, x - (new_w - w) // 2), max(0, y - (new_h - h) // 2)
    extracted_face = img[new_y:new_y + new_h, new_x:new_x + new_w]

    filename = os.path.join(os.getcwd(), output_path, "extracted_face.jpg")
    if os.path.exists(filename):
        os.remove(filename)

    cv2.imwrite(filename, extracted_face)
    logging.info(f"Extracted face saved at: {filename}")

    return filename


def deepface_face_comparison(image1_path, image2_path):
    logging.info("Verifying the images...")
    if not file_exists(image1_path) or not file_exists(image2_path):
        logging.warning("One or both image paths do not exist.")
        return False

    try:
        verification = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, enforce_detection=False)
        logging.debug(f"DeepFace verification result: {verification}")
        return verification.get("verified", False)
    except Exception as e:
        logging.exception(f"Face verification failed: {e}")
        return False


def get_face_embeddings(image_path):
    logging.info(f"Retrieving face embeddings from image: {image_path}")

    if not file_exists(image_path):
        logging.warning(f"Image path does not exist: {image_path}")
        return None

    try:
        embedding_objs = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        embedding = embedding_objs[0]["embedding"]
        logging.info("Face embeddings retrieved successfully")
        return embedding
    except Exception as e:
        logging.exception(f"Failed to retrieve embeddings: {e}")
        return None
