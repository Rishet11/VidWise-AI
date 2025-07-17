import streamlit as st
from ui.layout import build_layout
from ui.callbacks import handle_all_events

def main():
    st.set_page_config(page_title="VidWise AI", layout="centered")
    build_layout()
    handle_all_events()

if __name__ == "__main__":
    main()
