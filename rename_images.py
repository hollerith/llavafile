import os
import sys
import glob
import math
import json
import shutil
import subprocess
import pickle
import face_recognition
import logging

from datetime import datetime
from PIL import Image
from io import BytesIO
from ollama import generate
from logger_config import setup_logger

# Setup logger
logger = setup_logger()

# Function to read processed file names from all log files except the current one
def read_processed_files():
    processed_files = set()
    # Find all log files
    log_files = glob.glob('application_*.log')
    if log_files:
        # Sort log files by creation time and exclude the most recent one
        sorted_log_files = sorted(log_files, key=os.path.getctime)
        most_recent_log_file = sorted_log_files[-1]
        for log_file in sorted_log_files:
            # Skip the most recent log file
            if log_file == most_recent_log_file:
                continue
            with open(log_file, 'r') as log_file_content:
                for line in log_file_content:
                    if "Renaming from" in line:
                        old_name = line.split("Renaming from")[1].split("to")[0].strip()
                        processed_files.add(old_name)
    return processed_files

PROMPT = """
### User: Rename this image...
"""

def load_known_faces(known_faces_file):
    with open(known_faces_file, 'rb') as f:
        known_faces = pickle.load(f)
    return known_faces['encodings'], known_faces['names']

def get_image_files(folder_path, glob_pattern):
    if glob_pattern:
        return glob.glob(f"{folder_path}/{glob_pattern}")
    else:
        return glob.glob(f"{folder_path}/*.png") + glob.glob(f"{folder_path}/*.jpg")

def recognize_faces(image_file, known_face_encodings, known_face_names):
    image = face_recognition.load_image_file(image_file)
    unknown_face_encodings = face_recognition.face_encodings(image)
    recognized_names = []
    for unknown_face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            recognized_names.append(known_face_names[first_match_index])
    return recognized_names

def normalize_and_convert_image(image_file):
    _, file_extension = os.path.splitext(image_file)

    with Image.open(image_file) as full_image_path:
        width, height = full_image_path.size
        new_width = 320
        new_height = int(height * (new_width / width))
        full_image_path_resized = full_image_path.resize((new_width, new_height), Image.Resampling.LANCZOS)
        with BytesIO() as buffer:
            full_image_path_resized.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

    return image_bytes, file_extension

def shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in set(data):
        p_x = data.count(x) / len(data)
        entropy += - p_x * math.log2(p_x)
    return entropy

def generate_new_name(image_file, recognized_names):
    image_bytes, file_extension = normalize_and_convert_image(image_file)
    base_name = os.path.basename(image_file)

    prompt = f"{PROMPT}\nFilename: {base_name}\n"
    if recognized_names:
        prompt += ' ' + ', '.join(recognized_names)

    options = { 'temperature': 0 }
    message = generate(model='phlava:latest', prompt=prompt, images=[image_bytes], options=options, format='json', stream=False)

    if 'total_duration' in message:
        logging.info(f"Total duration for {base_name}: {message['total_duration']}")
    if 'eval_count' in message:
        logging.info(f"Evaluation count for {base_name}: {message['eval_count']}")

    response_json = json.loads(message['response'])
    try:
        filename = response_json['filename']
        description = response_json['description']
    except Exception as error:
        logging.error(f"ERROR: {error}")

    # Ensure the filename ends with file extension
    if not filename.endswith(file_extension):
        filename += file_extension.lower()

    return filename, description

def rename_images(source_dir, target_dir, glob_pattern, threshold=3.6):
    known_faces_file = 'known_faces.dat' # Hardcoded path to the known faces file
    known_face_encodings, known_face_names = load_known_faces(known_faces_file)

    # Read processed file names from log files
    processed_files = read_processed_files()

    for full_image_path in get_image_files(source_dir, glob_pattern):
        original_name = os.path.basename(full_image_path)

        # Skip if the file has already been processed
        if original_name in processed_files:
            logger.info(f"Skipping {full_image_path} as it has already been processed.")
            continue

        entropy = shannon_entropy(original_name)
        if entropy < threshold:
            logger.info(f"Skipping {full_image_path} as name entropy {entropy} less than {threshold}")
            continue

        recognized_names = recognize_faces(full_image_path, known_face_encodings, known_face_names)
        if recognized_names:
            logger.info(f"Known faces: {str(recognized_names)}")

        new_name, comment = generate_new_name(full_image_path, recognized_names)
        base_name = new_name
        new_full_path = os.path.join(target_dir, base_name)
        count = 1
        while os.path.exists(new_full_path):
            base_name = f"{new_name[:-4]}_{count}{new_name[-4:]}"
            new_full_path = os.path.join(target_dir, base_name)
            count += 1

        shutil.copy(full_image_path, new_full_path)

        # Extract month and year from the file's creation date
        creation_time = os.path.getctime(full_image_path)
        creation_date = datetime.fromtimestamp(creation_time)
        month_year = creation_date.strftime("%B %Y")

        # Add the old name, people, description, and month/year to the EXIF comment
        old_name = os.path.basename(full_image_path)
        logger.info(f"Renaming from {old_name} to {new_name}")

        comment = f"-Comment=Filename:{old_name} People:{', '.join(recognized_names)} Details: {comment} Month/Year: {month_year}"
        logger.info(f"Comment: {comment}")

        result = subprocess.run(['exiftool', comment, new_full_path, '-overwrite_original'], capture_output=True, text=True)
        logger.info(f"Exiftool: {base_name}: {result.stdout.strip()}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script_name.py <source_directory> <target_directory> <glob_pattern> [threshold]")
        sys.exit(1)

    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    glob_pattern = sys.argv[3]
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else None

    args = [source_dir, target_dir, glob_pattern]
    if threshold is not None:
        args.append(threshold)

    rename_images(*args)
