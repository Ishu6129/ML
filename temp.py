import cv2
import streamlit as st

def main():
    st.title("Camera Feed")

    # Attempt to open the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open the camera. Please check your device or permissions.")
        return

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()

if __name__ == "__main__":
    main()
