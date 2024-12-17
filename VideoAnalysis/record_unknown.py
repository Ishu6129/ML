import cv2
import face_recognition
import os
import glob
import numpy as np
import threading

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # Resize frame for faster speed

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_path)} encoding images found.")

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

        print("Encoding images loaded.")

    def detect_known_faces(self, frame):
        """Detect known faces in a frame"""
        # Resize frame for faster processing
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


class VideoCaptureThread(threading.Thread):
    def __init__(self, capture_device):
        super().__init__()
        self.cap = capture_device
        self.frame = None
        self.running = True

    def run(self):
        """Capture frames in a separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def stop(self):
        """Stop capturing frames"""
        self.running = False
        self.cap.release()

    def get_frame(self):
        """Get the latest captured frame"""
        return self.frame


def save_unknown_face(frame, face_location, face_id):
    """Save the image of the unknown face with a unique numeric ID"""
    y1, x2, y2, x1 = face_location
    unknown_face = frame[y1:y2, x1:x2]
    # Create the directory if it doesn't exist
    save_dir = "unknown_faces"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Save the face image with a unique numeric ID
    cv2.imwrite(f"{save_dir}/unknown_{face_id}.jpg", unknown_face)
    print(f"Captured unknown face as unknown_{face_id}.jpg")


def main():
    # Initialize face recognition
    sfr = SimpleFacerec()
    sfr.load_encoding_images("img_dataset/")

    # Open camera with DirectShow backend for better performance (Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Start video capture thread
    video_thread = VideoCaptureThread(cap)
    video_thread.start()

    print("Press 'A' to exit.")

    frame_count = 0
    unknown_face_id = 1  # Start with a unique ID for unknown faces

    while True:
        frame = video_thread.get_frame()
        if frame is None:
            continue

        # Process every 5th frame to reduce the load (adjust this number based on your performance)
        if frame_count % 5 == 0:
            # Detect faces in the frame
            face_locations, face_names = sfr.detect_known_faces(frame)

            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                # If the person is unknown, save their image with a unique numeric name
                if name == "Unknown":
                    save_unknown_face(frame, face_loc, unknown_face_id)
                    unknown_face_id += 1

            cv2.imshow("Frame", frame)

        # Increment frame count
        frame_count += 1

        key = cv2.waitKey(1)
        if key == ord('a'):  # Exit on pressing 'A'
            break

    # Stop video thread and release resources
    video_thread.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
