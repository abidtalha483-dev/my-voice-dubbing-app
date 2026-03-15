import streamlit as st
from pydub import AudioSegment
import os

st.title("AI Voice Changer")

uploaded_file = st.file_uploader("Audio file upload karein (WAV format recommended)", type=["mp3", "wav"])

if uploaded_file is not None:
    # File load karna
    sound = AudioSegment.from_file(uploaded_file)
    
    # Pitch control
    pitch = st.slider("Pitch adjust karein (Voice Change)", -5, 5, 0)
    
    # Pitch logic
    new_sample_rate = int(sound.frame_rate * (2.0 ** (pitch / 12.0)))
    new_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(44100)
    
    # Audio play aur download
    st.audio(new_sound.export(format="mp3"), format="audio/mp3")
    st.success("Voice Change ho gayi!")
