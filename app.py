import streamlit as st
import os
import subprocess
import whisper
from deep_translator import GoogleTranslator
import asyncio
import edge_tts

st.set_page_config(page_title="AI Video Dubber", page_icon="🎬", layout="wide")

st.title("🎬 AI Video Dubbing App (English to Hindi/Urdu)")
st.markdown("Upload an English video, and AI will dub it into Hindi/Urdu automatically!")

# Audio nikalne ka function
def extract_audio(video_path, audio_path):
    command =["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Nayi aawaz aur video jorne ka function
def merge_audio_video(video_path, audio_path, output_path):
    command =["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", output_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# AI aawaz banane ka function
async def generate_audio(text, output_path, voice="hi-IN-MadhurNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

# Video upload karne ka option
uploaded_video = st.file_uploader("Upload English Video (MP4) - Testing ke liye choti video (1-2 min) dalein", type=["mp4"])

if uploaded_video is not None:
    # Upload ki hui video ko save karna
    temp_video_path = "temp_input.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    st.video(temp_video_path)

    if st.button("🎙️ Start AI Dubbing (English to Hindi)"):
        with st.status("AI Video Process kar raha hai... Thora intezar karein!", expanded=True) as status:
            try:
                # Step 1: Extract Audio
                st.write("1️⃣ Video se aawaz alag ki ja rahi hai...")
                temp_audio_path = "temp_audio.wav"
                extract_audio(temp_video_path, temp_audio_path)

                # Step 2: Whisper Transcription
                st.write("2️⃣ Whisper AI aawaz sun kar English text likh raha hai...")
                model = whisper.load_model("tiny")
                result = model.transcribe(temp_audio_path)
                english_text = result["text"]
                st.info(f"**English Transcript:** {english_text}")

                # Step 3: Translation
                st.write("3️⃣ English ko Hindi/Urdu mein Translate kiya ja raha hai...")
                translator = GoogleTranslator(source='auto', target='hi')
                hindi_text = translator.translate(english_text)
                st.success(f"**Hindi Translation:** {hindi_text}")

                # Step 4: Text-to-Speech (Edge TTS)
                st.write("4️⃣ Nayi Hindi aawaz banai ja rahi hai...")
                dubbed_audio_path = "dubbed_audio.mp3"
                asyncio.run(generate_audio(hindi_text, dubbed_audio_path))

                # Step 5: Merging back to Video
                st.write("5️⃣ Nayi aawaz ko video par lagaya ja raha hai...")
                final_video_path = "final_dubbed_video.mp4"
                merge_audio_video(temp_video_path, dubbed_audio_path, final_video_path)

                status.update(label="✅ Video Successfully Dubbed!", state="complete", expanded=False)

                # Natija (Result) dikhana
                st.markdown("### 🎬 Aapki Dubbed Video Tayyar hai!")
                st.video(final_video_path)

                # Download karne ka button
                with open(final_video_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Dubbed Video",
                        data=file,
                        file_name="Dubbed_Movie_Clip.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"Processing mein ek error aagaya: {e}")
                status.update(label="❌ Error occurred", state="error")

st.markdown("---")
st.caption("Note: Yeh free server hai, isliye shuru mein sirf choti videos (1-2 minutes) daal kar test karein.")
