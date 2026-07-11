#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from google import genai
from config.settings import GOOGLE_API_KEY

def main():
    client = genai.Client(api_key=GOOGLE_API_KEY)
    print("Listing models:")
    for model in client.models.list():
        print(model.name)

if __name__ == "__main__":
    main()
