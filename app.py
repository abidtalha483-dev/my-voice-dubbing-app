import streamlit as st
import os
import subprocess
import whisper
from deep_translator import GoogleTranslator
import asyncio
import edge_tts
import librosa
import soundfile as sf
import numpy as np

st.set_page_config(page_title="Pro AI Video Dubber", page_icon="🌍", layout="wide")

st.title("🌍 Pro AI Video Dubbing & Lip-Sync App")
st.markdown("Upload a video, choose your language, and AI will dub it with perfect timing!")

# 🌍 Languages ki List (Translation Code aur Edge-TTS Voice ke sath)
SUPPORTED_LANGUAGES = {
    "Hindi": {"trans": "hi", "voice": "hi-IN-MadhurNeural"},
    "Urdu": {"trans": "ur", "voice": "ur-PK-AsadNeural"},
    "Spanish": {"trans": "es", "voice": "es-ES-AlvaroNeural"},
    "French": {"trans": "fr", "voice": "fr-FR-HenriNeural"},
    "Arabic": {"trans": "ar", "voice": "ar-SA-HamedNeural"},
    "Mandarin Chinese": {"trans": "zh-CN", "voice": "zh-CN-YunxiNeural"},
    "Bengali": {"trans": "bn", "voice": "bn-IN-BashkarNeural"},
    "Russian": {"trans": "ru", "voice": "ru-RU-DmitryNeural"},
    "Portuguese": {"trans": "pt", "voice": "pt-BR-AntonioNeural"},
    "Indonesian": {"trans": "id", "voice": "id-ID-ArdiNeural"},
    "Japanese": {"trans": "ja", "voice": "ja-JP-KeitaNeural"},
    "German": {"trans": "de", "voice": "de-DE-KillianNeural"},
    "Turkish": {"trans": "tr", "voice": "tr-TR-AhmetNeural"},
    "Tamil": {"trans": "ta", "voice": "ta-IN-PallaviNeural"},
    "Korean": {"trans": "ko", "voice": "ko-KR-InJoonNeural"},
    "Vietnamese": {"trans": "vi", "voice": "vi-VN-HoaiMyNeural"},
    "Italian": {"trans": "it", "voice": "it-IT-DiegoNeural"},
    "Polish": {"trans": "pl", "voice": "pl-PL-MarekNeural"},
    "Persian (Farsi)": {"trans": "fa", "voice": "fa-IR-FaridNeural"},
    "Dutch": {"trans": "nl", "voice": "nl-NL-MaartenNeural"},
    "Telugu": {"trans": "te", "voice": "te-IN-MohanNeural"},
    "Marathi": {"trans": "mr", "voice": "mr-IN-ManoharNeural"},
    "Gujarati": {"trans": "gu", "voice": "gu-IN-NiranjanNeural"},
    "Punjabi": {"trans": "pa", "voice": "pa-IN-OjasNeural"},
    "Malay": {"trans": "ms", "voice": "ms-MY-OsmanNeural"},
    "Thai": {"trans": "th", "voice": "th-TH-NiwatNeural"},
    "Greek": {"trans": "el", "voice": "el-GR-NestorasNeural"},
    "Ukrainian": {"trans": "uk", "voice": "uk-UA-OstapNeural"},
    "Swahili": {"trans": "sw", "voice": "sw-KE-ElimuNeural"}
}

# Helper Functions
def extract_audio(video_path, audio_path):
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def merge_audio_video(video_path, audio_path, output_path):
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", output_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

async def generate_audio(text, output_path, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

# UI Elements
selected_lang = st.selectbox("🌐 Konsi zaban (Language) mein dub karna hai?", list(SUPPORTED_LANGUAGES.keys()))
uploaded_video = st.file_uploader("Upload Video (MP4) - Choti video test karein", type=["mp4"])

if uploaded_video is not None:
    temp_video_path = "temp_input.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.video(temp_video_path)

    if st.button(f"🎙️ Start Dubbing in {selected_lang}"):
        with st.status(f"Processing Video to {selected_lang}... Please wait!", expanded=True) as status:
            try:
                # 1. Audio nikalna
                st.write("1️⃣ Extracting Original Audio...")
                temp_audio_path = "temp_audio.wav"
                extract_audio(temp_video_path, temp_audio_path)

                # 2. Whisper se AI Timestamps nikalna
                st.write("2️⃣ AI is analyzing timestamps (Lip-sync timing)...")
                model = whisper.load_model("tiny")
                result = model.transcribe(temp_audio_path)
                segments = result["segments"]

                # 3. Translation & Voice Generation setup
                st.write(f"3️⃣ Translating and matching voice timing for {selected_lang}...")
                target_trans = SUPPORTED_LANGUAGES[selected_lang]["trans"]
                target_voice = SUPPORTED_LANGUAGES[selected_lang]["voice"]
                translator = GoogleTranslator(source='auto', target=target_trans)

                # Khali Audio Canvas (Jis par hum dubbing rakhenge)
                sr = 24000
                final_audio = np.array([])

                # 4. Har line ko uske waqt par set karna (The Magic Loop)
                for seg in segments:
                    start_time = seg["start"]
                    end_time = seg["end"]
                    text = seg["text"].strip()
                    
                    if not text: continue
                    
                    # Translate
                    translated_text = translator.translate(text)
                    
                    # Nayi Aawaz banana
                    seg_audio_path = "temp_seg.mp3"
                    asyncio.run(generate_audio(translated_text, seg_audio_path, voice=target_voice))
                    
                    # Aawaz ki lambai (Speed) adjust karna
                    y_tts, _ = librosa.load(seg_audio_path, sr=sr)
                    target_dur = max(0.5, end_time - start_time) # Asli video mein banda kitni dair bola
                    actual_dur = librosa.get_duration(y=y_tts, sr=sr) # AI ne kitni dair mein bola
                    
                    if actual_dur > 0:
                        rate = actual_dur / target_dur
                        # Speed thori adjust karna taake natural lage
                        rate = max(0.7, min(rate, 1.5)) 
                        y_tts_stretched = librosa.effects.time_stretch(y_tts, rate=rate)
                    else:
                        y_tts_stretched = y_tts
                        
                    # Audio array mein fit karna
                    start_sample = int(start_time * sr)
                    end_sample = start_sample + len(y_tts_stretched)
                    
                    if len(final_audio) < end_sample:
                        final_audio = np.pad(final_audio, (0, end_sample - len(final_audio)))
                        
                    final_audio[start_sample:start_sample+len(y_tts_stretched)] = y_tts_stretched

                # 5. Final Synchronized Audio save karna
                st.write("4️⃣ Merging perfectly timed audio with video...")
                synced_audio_path = "dubbed_audio_sync.wav"
                sf.write(synced_audio_path, final_audio, sr)

                # Video ke sath jorna
                final_video_path = "final_dubbed_video.mp4"
                merge_audio_video(temp_video_path, synced_audio_path, final_video_path)

                status.update(label="✅ Dubbing Complete with Lip-Sync!", state="complete", expanded=False)

                st.markdown(f"### 🎬 Your {selected_lang} Dubbed Video is Ready!")
                st.video(final_video_path)

                with open(final_video_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Synced Video",
                        data=file,
                        file_name=f"Dubbed_{selected_lang}_Video.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"Error: {e}")
                status.update(label="❌ Error occurred", state="error")

st.markdown("---")
st.caption("AI Lip-Sync Enabled: Aawaz ab original time ke sath match hogi aur katega nahi!")
