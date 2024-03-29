import os
import shutil
import glob
import subprocess
from PIL import Image
from io import BytesIO
from ollama import generate
import face_recognition
import pickle
import math

PROMPT = """
### User: Make a filename for this image in six words or less including any names from this description.
The filename should be all lowercase with underscores instead of spaces and contain only alphabetic
characters. The goal is to make a reasonably short descriptive filename for the image.

Image description:
"""

def load_known_faces(known_faces_file):
    with open(known_faces_file, 'rb') as f:
        known_faces = pickle.load(f)
    return known_faces['encodings'], known_faces['names']

def get_image_files(folder_path):
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

def generate_new_name(image_file, recognized_names):
    with Image.open(image_file) as img:
        with BytesIO() as buffer:
            img.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

    # Llava just ignores prompts and describes the image,
    options = { 'temperature': 0 }
    llava = generate(model='llava:latest', prompt="Give a detailed description of the image", images=[image_bytes], options=options, stream=False)

    prompt = PROMPT + llava['response'][:500] + "### Assistant:"

    if recognized_names:
        prompt += ' ' + ', '.join(recognized_names)

    message = generate(model='mistral:latest', prompt=prompt, images=[image_bytes], options=options, stream=False)
    new_name = message['response'].strip().replace(' ', '_') + '.png'
    if not new_name.endswith('.png'):
        new_name += '.png'
    return new_name

def shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in set(data):
        p_x = data.count(x) / len(data)
        entropy += - p_x * math.log2(p_x)
    return entropy

def rename_images(source_dir):
    known_faces_file = 'known_faces.dat' # Hardcoded path to the known faces file
    known_face_encodings, known_face_names = load_known_faces(known_faces_file)
    for img in get_image_files(source_dir):
        original_name = os.path.basename(img)
        entropy = shannon_entropy(original_name)
        if entropy < 3.5:
            continue
        recognized_names = recognize_faces(img, known_face_encodings, known_face_names)
        new_name = generate_new_name(img, recognized_names)
        base_name = new_name
        count = 1
        while os.path.exists(base_name):
            base_name = f"{new_name[:-4]}_{count}{new_name[-4:]}"
            count += 1

        shutil.copy(img, base_name)

        # Add the old name as a comment to the new file using exiftool
        old_name = os.path.basename(img)
        subprocess.run(['exiftool', '-Comment=' + old_name, base_name, '-overwrite_original'])

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <source_directory>")
        sys.exit(1)

    source_dir = sys.argv[1]
    rename_images(source_dir)
