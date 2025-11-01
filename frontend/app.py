'''import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import requests

st.title("🎵 Real-Time Emotion-Based Music Recommendation")

BACKEND_URL = "http://localhost:8000/analyze/"

# Processor to capture frames
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_emotion = None
        self.playlists = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        _, img_encoded = cv2.imencode('.jpg', img)
        files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        try:
            response = requests.post(BACKEND_URL, files=files, timeout=30)
            data = response.json()
            self.last_emotion = data['emotion']
            self.playlists = data['playlists']
        except Exception as e:
            print("Error:", e)

        # Draw detected emotion on frame
        if self.last_emotion:
            cv2.putText(img, self.last_emotion, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img

webrtc_ctx = webrtc_streamer(
    key="emotion-detect",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Display playlists
if webrtc_ctx.video_processor:
    if webrtc_ctx.video_processor.playlists:
        st.subheader("🎶 Recommended Playlists:")
        for p in webrtc_ctx.video_processor.playlists:
            st.image(p['image'], width=150)
            st.markdown(f"[{p['name']}]({p['url']})")
'''

import streamlit as st
from PIL import Image
import requests

st.title("🎵 Emotion-Based Music Recommendation")

BACKEND_URL = "http://localhost:8000/analyze/"

# --- Choose input method ---
input_method = st.radio("Select Input Method:", ["Upload Image", "Capture from Webcam"])

image_bytes = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(Image.open(uploaded_file), caption="Uploaded Image", width=300)

elif input_method == "Capture from Webcam":
    captured_file = st.camera_input("Capture your image")
    if captured_file is not None:
        image_bytes = captured_file.read()
        st.image(Image.open(captured_file), caption="Captured Image", width=300)

# --- Send to backend ---
if image_bytes is not None:
    if st.button("Analyze Emotion"):
        try:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            response = requests.post(BACKEND_URL, files=files, timeout=30)
            data = response.json()
            
            emotion = data.get("emotion")
            playlists = data.get("playlists", [])

            st.success(f"Detected Emotion: {emotion}")

            if playlists:
                st.subheader("🎶 Recommended Playlists:")
                for p in playlists:
                    if not p or "image" not in p or not p["image"]:
                        continue
                    st.image(p["image"], width=150)
                    st.markdown(f"[{p['name']}]({p['url']})")
            else:
                st.info("No playlists found for this emotion.")

        except Exception as e:
            st.error(f"Error contacting backend: {e}")

