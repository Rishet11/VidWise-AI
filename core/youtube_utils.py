import re
import requests
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from config.secrets import SCRAPERAPI_KEY
import streamlit as st

def extract_youtube_id(url):
    parsed = urlparse(url)

    # Case 1: youtu.be short URL
    if parsed.netloc == "youtu.be":
        return parsed.path.lstrip("/")

    # Case 2: youtube.com/watch?v=...
    if "youtube.com" in parsed.netloc:
        query_params = parse_qs(parsed.query)
        if "v" in query_params:
            return query_params["v"][0]

        # Case 3: /embed/VIDEO_ID
        match_embed = re.search(r"/embed/([a-zA-Z0-9_-]{11})", parsed.path)
        if match_embed:
            return match_embed.group(1)

        # Case 4: /shorts/VIDEO_ID
        match_shorts = re.search(r"/shorts/([a-zA-Z0-9_-]{11})", parsed.path)
        if match_shorts:
            return match_shorts.group(1)

    return None 


@st.cache_data(show_spinner="📄 Fetching transcript...")
def get_transcript(video_id):
    proxy_url = f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"

    session = requests.Session()
    session.proxies = {
        "http": proxy_url,
        "https": proxy_url
    }
    
    session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
    })

    # In newer versions of youtube-transcript-api, we instantiate the API with the session
    api = YouTubeTranscriptApi(http_client=session)
    
    # Optional: diagnostic check
    try:
        ip_check = session.get("http://httpbin.org/ip")
        print("Detected IP via ScraperAPI:", ip_check.text)
    except:
        pass

    try:
        # fetch() returns a list of dictionaries with 'text', 'start', 'duration'
        transcript_list = api.fetch(video_id)
        return " ".join([d["text"] for d in transcript_list])
    
    except Exception as e1:
        print(f"Primary fetch failed: {e1}")
        try:
            transcript_list = api.fetch(video_id, languages=["hi", "en"])
            return " ".join([d["text"] for d in transcript_list])
        except Exception as e2:
            raise RuntimeError(f"❌ Could not fetch transcript: {e2}")