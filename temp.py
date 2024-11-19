import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer

def main():
    st.title("OpenCV Camera Feed with Streamlit")
    st.write("This app displays the live camera feed using OpenCV.")

    # Start the WebRTC streamer
    webrtc_streamer(key="camera")

if __name__ == "__main__":
    main()
