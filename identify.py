import face_recognition
import pickle
import sys

with open('known_faces.dat', 'rb') as data_file:
    known_data = pickle.load(data_file)
known_face_encodings = known_data['encodings']
known_face_names = known_data['names']

if len(sys.argv) < 1:
    print('None')
    sys.exit(1)

image_path = sys.argv[1]
unknown_image = face_recognition.load_image_file(image_path)
unknown_face_encodings = face_recognition.face_encodings(unknown_image)

if len(unknown_face_encodings) == 0:
    sys.exit(2)

recognized_names = []

for unknown_face_encoding in unknown_face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
    best_match_index = None if len(face_distances) == 0 else face_distances.argmin()

    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    recognized_names.append(name)

if recognized_names:
    print(", ".join(set(recognized_names)) + "\n")
else:
    print("None")
    sys.exit(0)
