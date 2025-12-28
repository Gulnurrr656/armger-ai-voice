import os
import json
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import texttospeech
from google.oauth2 import service_account

# ---------- Google Credentials ----------
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in os.environ:
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON is not set")

creds_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
credentials = service_account.Credentials.from_service_account_info(creds_info)

tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

# ---------- FastAPI ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schema ----------
class SpeakRequest(BaseModel):
    text: str
    language_code: str = "ru-RU"
    voice_name: str = "ru-RU-Wavenet-B"

# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/speak")
def speak(data: SpeakRequest):
    if not data.text:
        raise HTTPException(status_code=400, detail="Text is empty")

    synthesis_input = texttospeech.SynthesisInput(text=data.text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=data.language_code,
        name=data.voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")

    return {
        "audio": audio_base64
    }

# alias
@app.post("/voice")
def voice(data: SpeakRequest):
    return speak(data)
