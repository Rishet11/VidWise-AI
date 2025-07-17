import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
SCRAPERAPI_KEY = get_secret("SCRAPERAPI_KEY")
