from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from emotion_detector import predict_emotion
from spotify_recommender import get_playlist_for_emotion

app = FastAPI()

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze_emotion(text: str = Form(...), file: UploadFile = File(...)):
    try:
        payload = text.strip()
        contents = await file.read() if file else b""
        emotion = predict_emotion(contents, payload)
        playlists = get_playlist_for_emotion(emotion)
        return {"emotion": emotion, "playlists": playlists}
    except Exception as e:
        print("Error in /analyze/:", e)
        return {"emotion": "Error", "playlists": [], "error": str(e)}