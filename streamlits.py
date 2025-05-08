import streamlit as st

st.title("Live CCTV Stream dengan YOLOv8")

st.components.v1.html(
    '<img src="http://localhost:5000/video_feed" width="720">',
    height=500
)
