import streamlit as st
import numpy as np
import soundfile as sf
import tempfile
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🎤 Voice Chatbot", layout="centered")
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- UI ----------------
st.title("🎤 Voice-Enabled Chatbot")
st.caption("Speak → AI Thinks → AI Talks")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- VOICE INPUT ----------------
audio = mic_recorder(
    start_prompt="🎙️ Speak",
    stop_prompt="⏹️ Stop",
    key="mic"
)

def speech_to_text(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_file = open(f.name, "rb")

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file
    )
    return transcript.text

def chat_with_ai(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content

def text_to_speech(text):
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )
    audio = np.frombuffer(response.audio, dtype=np.int16)
    return audio

# ---------------- PROCESS AUDIO ----------------
if audio:
    user_text = speech_to_text(audio["bytes"])

    st.session_state.messages.append(
        {"role": "user", "content": user_text}
    )

    with st.chat_message("user"):
        st.write(user_text)

    ai_reply = chat_with_ai(st.session_state.messages)

    st.session_state.messages.append(
        {"role": "assistant", "content": ai_reply}
    )

    with st.chat_message("assistant"):
        st.write(ai_reply)

    audio_reply = text_to_speech(ai_reply)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_reply, 24000)
        st.audio(f.name)
