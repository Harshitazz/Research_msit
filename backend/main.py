from fastapi import FastAPI, File, UploadFile
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
async def analyze_emotion(file: UploadFile = File(...)):
    contents = await file.read()
    emotion = predict_emotion(contents)
    playlists = get_playlist_for_emotion(emotion)
    return {"emotion": emotion, "playlists": playlists}
