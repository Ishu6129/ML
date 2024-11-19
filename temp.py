import cv2
import streamlit as st

# Function to initialize and check the camera
def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Error: Could not access the camera. Please ensure the camera is connected.")
        return None
    return cap

# Function to display the webcam feed
def display_video_stream(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture image from camera.")
            break

        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        # Close the video stream on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Streamlit app interface
def main():
    st.title("Webcam Stream using OpenCV and Streamlit")

    # Select camera index (0 is default, 1 or 2 can be used for multiple cameras)
    camera_index = st.selectbox("Select Camera", [0, 1, 2])

    # Initialize the camera
    cap = initialize_camera(camera_index)

    if cap:
        display_video_stream(cap)

        # Release the camera when done
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
