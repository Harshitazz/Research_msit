import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
    client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET")
))

# Example emotion-to-genre mapping
EMOTION_GENRE_MAP = {
    "Happy": "party",
    "Sad": "acoustic",
    "Angry": "rock",
    "Fear": "ambient",
    "Surprise": "pop",
    "Neutral": "chill",
    "Disgust": "alternative"
}

def get_playlist_for_emotion(emotion):
    """Fetch Spotify playlists based on detected emotion."""
    genre = EMOTION_GENRE_MAP.get(emotion, "pop")

    try:
        # 1️⃣ Try searching playlists directly
        results = sp.search(q=genre, type='playlist', limit=5)
        items = results.get("playlists", {}).get("items", [])
        playlists = []

        # 2️⃣ If items are empty or None, use browse endpoint instead
        if not any(items):
            print(f"No valid playlists found in search for '{genre}', using featured playlists...")
            featured = sp.categories_playlists(category_id="toplists", country="US", limit=5)
            items = featured.get("playlists", {}).get("items", [])

        print(items)

        # 3️⃣ Process playlists safely
        for playlist in items:
            if not playlist:
                continue
            playlists.append({
                "name": playlist.get("name", "Unknown Playlist"),
                "url": playlist.get("external_urls", {}).get("spotify", "#"),
                "image": (playlist.get("images", [{}])[0].get("url")
                          if playlist.get("images") else None)
            })

        
        return playlists

    except Exception as e:
        print(f"Error fetching playlists for emotion {emotion}: {e}")
        # Final fallback
        return [
            {
                "name": "Fallback Chill Mix",
                "url": "https://open.spotify.com/playlist/37i9dQZF1DX4WYpdgoIcn6",
                "image": "https://i.scdn.co/image/ab67706f00000002d27dcb1e3bb73053f17e0d0d"
            }
        ]