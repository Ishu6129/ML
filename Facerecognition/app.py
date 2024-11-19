import streamlit as st
import cv2
import face_recognition
import os
import glob
import numpy as np
from PIL import Image


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # Resize frame for faster speed

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename from the image path
            basename = os.path.basename(img_path)
            filename, ext = os.path.splitext(basename)

            # Get face encoding
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:  # Only add encoding if valid
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(filename)

    def detect_known_faces(self, frame):
        """Detect known faces in a frame"""
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            else:
                name = "Unknown"

            face_names.append(name)

        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names


# Streamlit App
def main():
    st.title("Face Recognition App")

    # Initialize face recognition
    sfr = SimpleFacerec()
    sfr.load_encoding_images("img_dataset/")  # Folder with stored images
    st.sidebar.header("Face Recognition Settings")
    st.sidebar.write(f"Loaded {len(sfr.known_face_names)} known faces.")

    # Select an image file
    uploaded_file = st.file_uploader("Upload an image for face detection:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Detect faces in the uploaded image
        face_locations, face_names = sfr.detect_known_faces(frame)

        # Annotate the image
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Convert the image to RGB for displaying with PIL
        annotated_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(annotated_image, caption="Processed Image", use_column_width=True)


if __name__ == "__main__":
    main()
