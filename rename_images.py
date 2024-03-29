import os
import shutil
import glob
import subprocess
import logging

from PIL import Image
from io import BytesIO
from ollama import generate
import face_recognition
import pickle
import math
import json

# ANSI escape sequences for different colors
class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    white = "\x1b[37;20m"

    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(f"{self.grey}%(asctime)s{self.reset} - {log_fmt}%(levelname)s{self.reset} - {self.white}%(message)s{self.reset}", "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# Configure logging with color
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

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
    # Extract the file extension
    _, file_extension = os.path.splitext(image_file)

    with Image.open(image_file) as img:
        width, height = img.size
        new_width = 320
        new_height = int(height * (new_width / width))
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        with BytesIO() as buffer:
            img_resized.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

    return image_bytes, file_extension

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
        filename += file_extension

    return filename, description

def shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in set(data):
        p_x = data.count(x) / len(data)
        entropy += - p_x * math.log2(p_x)
    return entropy

def rename_images(source_dir, glob_pattern):
    known_faces_file = 'known_faces.dat' # Hardcoded path to the known faces file
    known_face_encodings, known_face_names = load_known_faces(known_faces_file)
    for img in get_image_files(source_dir, glob_pattern):
        original_name = os.path.basename(img)
        entropy = shannon_entropy(original_name)
        if entropy < 3.5:
            continue
        recognized_names = recognize_faces(img, known_face_encodings, known_face_names)
        if recognized_names:
            logging.info(f"Known faces: {str(recognized_names)}")

        new_name, comment = generate_new_name(img, recognized_names)
        base_name = new_name
        count = 1
        while os.path.exists(base_name):
            base_name = f"{new_name[:-4]}_{count}{new_name[-4:]}"
            count += 1

        shutil.copy(img, base_name)

        # Add the old name, people and description as a comment to the new file using exiftool
        old_name = os.path.basename(img)
        logging.info(f"Renaming from {old_name} to {new_name}")

        comment = f"-Comment=Filename:{old_name} People:{', '.join(recognized_names)} Details: {comment}"
        logging.info(f"Comment: {comment}")

        result = subprocess.run(['exiftool', comment, base_name, '-overwrite_original'], capture_output=True, text=True)
        logging.info(f"Exiftool: {base_name}: {result.stdout.strip()}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script_name.py <source_directory> <glob_pattern>")
        sys.exit(1)

    source_dir = sys.argv[1]
    glob_pattern = sys.argv[2]
    rename_images(source_dir, glob_pattern)
