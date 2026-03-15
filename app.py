import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import io

st.set_page_config(page_title="Voice Changer App", page_icon="🎤")

st.title("🎤 AI Voice Changer & Dubbing App")
st.markdown("Python 3.14+ Compatible | Powered by Librosa")

# File uploader
uploaded_file = st.file_uploader("Apni audio file upload karein (WAV ya MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    st.markdown("### 🎛️ Audio Settings")
    
    # Sliders for Pitch and Speed
    pitch_steps = st.slider("Pitch (Aawaz ki Shruti)", min_value=-12.0, max_value=12.0, value=0.0, step=0.5, 
                            help="-ve means aawaz bhari hogi, +ve means patli hogi")
    speed_rate = st.slider("Speed (Raftaar)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    if st.button("Apply Changes"):
        with st.spinner("Processing audio... Please wait!"):
            try:
                # 1. Audio load karna (Librosa mp3 aur wav dono handle kar lega)
                # sr=None rakhne se original sample rate maintain rehta hai
                y, sr = librosa.load(uploaded_file, sr=None)
                
                # 2. Pitch change karna
                if pitch_steps != 0.0:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
                
                # 3. Speed change karna
                if speed_rate != 1.0:
                    y = librosa.effects.time_stretch(y, rate=speed_rate)
                
                # 4. Processed audio ko memory mein save karna (Virtual File)
                buffer = io.BytesIO()
                sf.write(buffer, y, sr, format='WAV')
                buffer.seek(0)
                
                st.success("Audio successfully process ho gayi!")
                
                # 5. Playback aur Download ka option dena
                st.audio(buffer, format="audio/wav")
                st.download_button(
                    label="⬇️ Download Processed Audio",
                    data=buffer,
                    file_name="voice_changed_audio.wav",
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"Ek error aagaya: {e}")

st.markdown("---")
st.caption("🚀 Future Scaling: Yeh architecture RVC aur Whisper (AI Dubbing) ke liye completely ready hai!")
