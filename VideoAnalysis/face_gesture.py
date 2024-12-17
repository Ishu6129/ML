import os
import cv2
import mediapipe as mp
import pyttsx3
import threading
import platform
from simple_facerec import SimpleFacerec
import streamlit as st
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()
sfr = SimpleFacerec()
sfr.load_encoding_images("img_dataset/")

def play_beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
    else:
        os.system('echo -e "\a"')  

# Text-to-speech function
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Gesture Detection
def detect_gesture(landmarks):
    thumb_tip = landmarks[4]  # Thumb Tip
    index_tip = landmarks[8]  # Index Tip
    middle_tip = landmarks[12]  # Middle Tip
    ring_tip = landmarks[16]  # Ring Tip
    pinky_tip = landmarks[20]  # Pinky Tip
    
    # Check for "Good" gesture (open hand with fingers spread)
    if thumb_tip.y > index_tip.y and middle_tip.y > index_tip.y:  # Example for 'Good'
        return "Good"
    
    # Check for "Danger" gesture (index finger up, others curled)
    elif thumb_tip.y < index_tip.y and middle_tip.y > index_tip.y:  # Example for 'Danger'
        return "Danger"
    
    # Check for "Critical" gesture (fist, all fingers curled)
    elif (index_tip.y > middle_tip.y and middle_tip.y > ring_tip.y and ring_tip.y > pinky_tip.y):  # Fist gesture
        return "Critical"
    
    return "None"

# Danger beep handler
def danger_beep():
    while danger_active:
        play_beep()
        time.sleep(0.5)

# Critical situation handler
def critical_emergency():
    speak_text("Emergency")

# Initialize global variables
danger_active = False
critical_active = False

# Capture face in a separate thread
def capture_face(frame, user_name):
    folder_path = os.path.join("img_dataset")
    os.makedirs(folder_path, exist_ok=True)
    img_path = os.path.join(folder_path, f"{user_name}.jpg")
    cv2.imwrite(img_path, frame)  # Save the current frame as the user's image
    st.success(f"Face captured and registered for {user_name}!")
    sfr.load_encoding_images("img_dataset/")  # Reload encodings with new image

# Main application function
def main():
    global danger_active, critical_active

    st.title("Real-Time Gesture & Face Recognition")

    # Layout buttons in a single row
    col1, col2, col3 = st.columns(3)
    with col1:
        start_button = st.button("Start Video")
    with col2:
        stop_button = st.button("Stop Video")
    with col3:
        capture_button = st.button("Capture and Register Face")

    # Session state for controlling video
    if "video_running" not in st.session_state:
        st.session_state.video_running = False

    # Handle Start and Stop actions
    if start_button:
        st.session_state.video_running = True
    if stop_button:
        st.session_state.video_running = False

    # Start video processing if enabled
    if st.session_state.video_running:
        cap = cv2.VideoCapture(0)

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Allow multiple hands
            min_detection_confidence=0.5,  # Lower confidence for distant detection
            min_tracking_confidence=0.5   # Lower tracking confidence
        ) as hands:
            # Stream video frames
            frame_placeholder = st.empty()  # Placeholder for displaying video frames

            while st.session_state.video_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video. Please check your webcam.")
                    break

                # Flip and process frame
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # **Scale for better distance detection**
                height, width, _ = frame.shape
                new_size = (int(width * 1.5), int(height * 1.5))  # Scale up
                frame_rgb = cv2.resize(frame_rgb, new_size)

                # Gesture recognition
                gesture = "None"
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        gesture = detect_gesture(hand_landmarks.landmark)
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Gesture actions
                if gesture == "Danger" and not danger_active:
                    danger_active = True
                    critical_active = False
                    threading.Thread(target=danger_beep, daemon=True).start()

                elif gesture == "Critical" and not critical_active:
                    critical_active = True
                    danger_active = False
                    threading.Thread(target=critical_emergency, daemon=True).start()

                elif gesture == "Good":
                    danger_active = False
                    critical_active = False

                # Face recognition
                face_locations, face_names = sfr.detect_known_faces(frame)
                for face_loc, name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display gesture on frame
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Resize back to original size for rendering
                frame = cv2.resize(frame, (width, height))

                # Render the video frame in Streamlit
                frame_placeholder.image(frame, channels="BGR", use_column_width=True)

                # Capture face registration
                if capture_button:
                    user_name = st.text_input("Enter your name:")
                    if user_name:
                        threading.Thread(target=capture_face, args=(frame, user_name), daemon=True).start()
                        break

            cap.release()
        st.info("Video Stream Stopped.")

# Run the application
if __name__ == "__main__":
    main()
