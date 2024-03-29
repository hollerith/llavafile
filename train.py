import face_recognition
import pickle
import os

# Directory containing folders of known individuals
known_faces_dir = './known_faces'

known_face_encodings = []
known_face_names = []

for name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, name)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)
        # Load each image
        image = face_recognition.load_image_file(image_path)
        # Try to find face encodings within the image
        try:
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
        except IndexError:
            print(f"No face found in {image_path}, skipping.")

# Save the face encodings and names to disk
with open('known_faces.dat', 'wb') as f:
    pickle.dump({'encodings': known_face_encodings, 'names': known_face_names}, f)

print("Training complete, data saved to known_faces.dat")

