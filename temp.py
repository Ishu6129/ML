import cv2
import streamlit as st
from PIL import Image

def main():
    st.title("OpenCV Camera Feed with Streamlit")

    # Open the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open the camera.")
        return

    stframe = st.empty()

    # Add a "Stop" button to exit the loop
    stop_button = st.button("Stop Camera")

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
            break

        # Convert frame (BGR to RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame, channels="RGB", use_column_width=True)

    # Release the camera when done
    cap.release()
    st.success("Camera stopped.")

if __name__ == "__main__":
    main()
