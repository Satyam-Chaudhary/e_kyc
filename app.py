import os
import logging
import streamlit as st
from preprocess import read_image, extract_id_card, save_image
from ocr_engine import extract_text
from postprocess import extract_information, extract_information1
from face_verification import detect_and_extract_face, deepface_face_comparison, get_face_embeddings
from sql_connection import insert_records, fetch_records, check_duplicacy, insert_records_aadhar, fetch_records_aadhar, check_duplicacy_aadhar
import toml
import hashlib
import pymysql
from datetime import datetime

pymysql.install_as_MySQLdb()

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

config = toml.load("config.toml")
db_config = config.get("database", {})

db_user = db_config.get("user")
db_password = db_config.get("password")


def hash_id(id_value):
    hash_object = hashlib.sha256(id_value.encode())
    return hash_object.hexdigest()


# Convert DOB to 'YYYY-MM-DD'
def normalize_dob(dob_value):
    if isinstance(dob_value, datetime):
        return dob_value.strftime('%Y-%m-%d')
    elif isinstance(dob_value, str):
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(dob_value, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return dob_value
    return dob_value


def wider_page():
    max_width_str = "max-width: 1200px;"
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{ {max_width_str} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_custom_theme():
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6;
                color: #333333;
            }
            .sidebar .sidebar-content {
                background-color: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_section():
    st.sidebar.title("Select ID Card Type")
    return st.sidebar.selectbox("Document Type", ("PAN", "AADHAR"))


def header_section(option):
    st.title("Registration Using Aadhar Card" if option == "AADHAR" else "Registration Using PAN Card")


def main_content(image_file, face_image_file, option):
    if image_file is None:
        st.warning("Please upload an ID card image.")
        return

    face_image = read_image(face_image_file, is_uploaded=True)
    if face_image is None:
        st.error("Face image not uploaded. Please upload a face image.")
        return

    image = read_image(image_file, is_uploaded=True)
    image_roi, _ = extract_id_card(image)

    face_image_path2 = detect_and_extract_face(img=image_roi)
    if face_image_path2 is None:
        st.error("No face detected in ID card. Please try again.")
        return

    face_image_path1 = save_image(face_image, "face_image.jpg", path="data/02_intermediate_data")
    is_face_verified = deepface_face_comparison(image1_path=face_image_path1, image2_path=face_image_path2)

    if not is_face_verified:
        st.error("Face verification failed. Please try again.")
        return

    extracted_text = extract_text(image_roi)
    text_info = extract_information(extracted_text) if option == "PAN" else extract_information1(extracted_text)
    text_info['ID'] = hash_id(text_info['ID'])

    records = fetch_records(text_info) if option == "PAN" else fetch_records_aadhar(text_info)
    if records.shape[0] > 0:
        st.write(records.shape)
        st.write(records)

    is_duplicate = check_duplicacy(text_info) if option == "PAN" else check_duplicacy_aadhar(text_info)
    if is_duplicate:
        st.write(f"User already present with ID {text_info['ID']}")
        return

    text_info['DOB'] = normalize_dob(text_info['DOB'])
    text_info['Embedding'] = get_face_embeddings(face_image_path1)

    insert_records(text_info) if option == "PAN" else insert_records_aadhar(text_info)
    st.success("User registered successfully.")

    result_to_display = text_info.copy()
    if "Embedding" in result_to_display:
        del result_to_display["Embedding"]

    st.json(result_to_display)


def main():
    st.connection(
        "local_db",
        type="sql",
        url=f"mysql://{db_user}:{db_password}@localhost:3306/ekyc"
    )

    wider_page()
    set_custom_theme()

    option = sidebar_section()
    header_section(option)

    image_file = st.file_uploader("Upload ID Card")
    if image_file:
        face_image_file = st.file_uploader("Upload Face Image")
        if face_image_file:
            _ = main_content(image_file, face_image_file, option)


if __name__ == "__main__":
    main()
