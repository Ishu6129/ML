import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self, frame_resizing=0.25):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = frame_resizing

    def load_encoding_images(self, images_path):
        """
        Load and encode images from a specified directory.
        """
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_path)} encoding images found.")

        for img_path in images_path:
            # Load image and convert it to RGB
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}. Skipping...")
                continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract filename (without extension)
            filename = os.path.splitext(os.path.basename(img_path))[0]

            # Get encoding, if available
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(filename)
            else:
                print(f"No face found in {img_path}. Skipping...")

        print("Encoding images loaded successfully.")

    def detect_known_faces(self, frame):
        """
        Detect and recognize faces in the given frame.
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare face with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            # Determine the best match
            name = "Unknown"
            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Adjust face locations back to the original frame size
        face_locations = (np.array(face_locations) / self.frame_resizing).astype(int)
        return face_locations, face_names
