import streamlit as st
import whisper
import subprocess
import os
import librosa
import numpy as np
import soundfile as sf
import asyncio
import edge_tts
from deep_translator import GoogleTranslator
import tempfile

st.set_page_config(page_title="AI Video Dubber Pro", page_icon="🎬", layout="wide")

st.title("🎬 AI Video Dubbing App (Pro Version)")
st.markdown("Upload a video, choose your language, and AI will dub it with Lip-Sync matching!")

uploaded_file = st.file_uploader("Upload MP4 Video (Testing ke liye 1-2 min ki video dalein)", type=["mp4"])

LANGUAGES = {
"Mandarin Chinese": ("zh-CN","zh-CN-XiaoxiaoNeural"),
"Hindi": ("hi","hi-IN-SwaraNeural"),
"Spanish": ("es","es-ES-ElviraNeural"),
"French": ("fr","fr-FR-DeniseNeural"),
"Arabic": ("ar","ar-SA-ZariyahNeural"),
"Bengali": ("bn","bn-BD-NabanitaNeural"),
"Russian": ("ru","ru-RU-SvetlanaNeural"),
"Portuguese": ("pt","pt-BR-FranciscaNeural"),
"Urdu": ("ur","ur-PK-UzmaNeural"),
"Indonesian": ("id","id-ID-GadisNeural"),
"Japanese": ("ja","ja-JP-NanamiNeural"),
"German": ("de","de-DE-KatjaNeural"),
"Swahili": ("sw","sw-KE-ZuriNeural"),
"Turkish": ("tr","tr-TR-EmelNeural"),
"Tamil": ("ta","ta-IN-PallaviNeural"),
"Marathi": ("mr","mr-IN-AarohiNeural"),
"Telugu": ("te","te-IN-ShrutiNeural"),
"Korean": ("ko","ko-KR-SunHiNeural"),
"Vietnamese": ("vi","vi-VN-HoaiMyNeural"),
"Italian": ("it","it-IT-ElsaNeural"),
"Punjabi": ("pa","pa-IN-GaganNeural"),
"Persian (Farsi)": ("fa","fa-IR-DilaraNeural"),
"Gujarati": ("gu","gu-IN-DhwaniNeural"),
"Malay": ("ms","ms-MY-YasminNeural"),
"Thai": ("th","th-TH-PremwadeeNeural"),
"Kannada": ("kn","kn-IN-SapnaNeural"),
"Malayalam": ("ml","ml-IN-SobhanaNeural"),
"Dutch": ("nl","nl-NL-ColetteNeural"),
"Greek": ("el","el-GR-AthinaNeural"),
"Ukrainian": ("uk","uk-UA-PolinaNeural"),
"Polish": ("pl","pl-PL-ZofiaNeural"),
"Romanian": ("ro","ro-RO-AlinaNeural"),
"Czech": ("cs","cs-CZ-VlastaNeural"),
"Hungarian": ("hu","hu-HU-NoemiNeural"),
"Hebrew": ("he","he-IL-HilaNeural"),
"Bulgarian": ("bg","bg-BG-KalinaNeural"),
"Slovak": ("sk","sk-SK-ViktoriaNeural"),
"Finnish": ("fi","fi-FI-NooraNeural"),
"Danish": ("da","da-DK-ChristelNeural"),
"Norwegian": ("no","nb-NO-PernilleNeural"),
"Swedish": ("sv","sv-SE-SofieNeural"),
"Lithuanian": ("lt","lt-LT-OnaNeural"),
"Latvian": ("lv","lv-LV-EveritaNeural"),
"Estonian": ("et","et-EE-AnuNeural"),
"Tagalog (Filipino)": ("tl","fil-PH-BlessicaNeural"),
"Nepali": ("ne","ne-NP-HemkalaNeural"),
"Pashto": ("ps","ps-AF-LatifaNeural"),
"Amharic": ("am","am-ET-MekdesNeural"),
"Zulu": ("zu","zu-ZA-ThandoNeural"),
"Xhosa": ("xh","xh-ZA-ThandoNeural"),
"Yoruba": ("yo","yo-NG-AdetounNeural"),
"Igbo": ("ig","ig-NG-EzinneNeural")
}

target_language = st.selectbox("🌍 Select Target Language (Kis zaban mein dub karna hai?)", list(LANGUAGES.keys()))

async def generate_tts(text, voice, path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)

def extract_audio(video, audio):
    cmd =["ffmpeg", "-y", "-i", video, "-ac", "1", "-ar", "16000", audio]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def merge_audio(video, audio, output):
    cmd =["ffmpeg", "-y", "-i", video, "-i", audio, "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", output]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if uploaded_file:
    # Display the uploaded video
    st.video(uploaded_file)

    if st.button("🎙️ Start Dubbing (With Lip-Sync)"):
        with st.status("🚀 Processing Video... (Isme thora time lag sakta hai)", expanded=True) as status:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    video_path = os.path.join(tmp, "video.mp4")
                    audio_path = os.path.join(tmp, "audio.wav")

                    # Save uploaded file
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.read())

                    st.write("1️⃣ Extracting audio from video...")
                    extract_audio(video_path, audio_path)

                    st.write("2️⃣ Loading Whisper AI (Understanding Speech)...")
                    model = whisper.load_model("tiny")

                    st.write("3️⃣ Transcribing and finding lip-sync timings...")
                    result = model.transcribe(audio_path)
                    segments = result["segments"]

                    translator_code, voice = LANGUAGES[target_language]

                    # Load original audio to get its length
                    y_original, sr = librosa.load(audio_path, sr=16000)
                    
                    # Create a blank canvas of the same length to paste new audio on it
                    canvas = np.zeros(len(y_original))

                    st.write(f"4️⃣ Translating & Generating {target_language} Voice...")
                    
                    # Process segment by segment for Lip Sync
                    for i, seg in enumerate(segments):
                        start = seg["start"]
                        end = seg["end"]
                        text = seg["text"]

                        # Translate
                        translated = GoogleTranslator(source='auto', target=translator_code).translate(text)

                        # Generate Audio
                        tts_path = os.path.join(tmp, f"tts_{i}.mp3")
                        asyncio.run(generate_tts(translated, voice, tts_path))

                        # Load Generated Audio
                        y_tts, _ = librosa.load(tts_path, sr=16000)

                        # Calculate Timings
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        
                        # Prevent Array Out of Bound error
                        if start_sample >= len(canvas):
                            continue
                        if end_sample > len(canvas):
                            end_sample = len(canvas)

                        target_len = end_sample - start_sample
                        
                        if target_len > 0:
                            # Adjust Speed (Time Stretch) to match lip sync
                            rate = len(y_tts) / target_len
                            y_match = librosa.effects.time_stretch(y_tts, rate=rate)

                            # Fit perfectly into the required length
                            if len(y_match) > target_len:
                                y_match = y_match[:target_len]
                            else:
                                pad = target_len - len(y_match)
                                y_match = np.pad(y_match, (0, pad))

                            # Put the audio on the canvas at the exact time
                            canvas[start_sample:end_sample] += y_match

                    st.write("5️⃣ Merging new audio back to video...")
                    final_audio = os.path.join(tmp, "dubbed.wav")
                    sf.write(final_audio, canvas, sr)

                    final_video = "final_output.mp4"
                    merge_audio(video_path, final_audio, final_video)

                    status.update(label="✅ Video Successfully Dubbed!", state="complete", expanded=False)

                    # Show Final Video
                    st.success("🎉 Aapki Video Tayyar Hai!")
                    st.video(final_video)

                    with open(final_video, "rb") as f:
                        st.download_button("⬇️ Download Dubbed Video", f, file_name="dubbed_video_lipsync.mp4", mime="video/mp4")

            except Exception as e:
                st.error(f"❌ Ek error aagaya: {e}")
                status.update(label="Error occurred", state="error")
