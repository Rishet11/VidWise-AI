#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from google import genai
from config.settings import GOOGLE_API_KEY

def main():
    client = genai.Client(api_key=GOOGLE_API_KEY)
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite"
    ]
    for model in models_to_try:
        try:
            print(f"Trying model: {model}...")
            response = client.models.generate_content(
                model=model,
                contents="Hello, say 'yes' if you work.",
            )
            print(f"  Success! Response: {response.text.strip()}")
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    main()
