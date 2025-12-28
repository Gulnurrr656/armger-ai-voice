import os
import json
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import texttospeech
from google.oauth2 import service_account

# ================== GOOGLE TTS ==================

if "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in os.environ:
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON is not set")

creds_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
credentials = service_account.Credentials.from_service_account_info(creds_info)

tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

# ================== APP ==================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== SCHEMAS ==================

class AskRequest(BaseModel):
    question: str

# ================== HELPERS ==================

def speak_text(text: str) -> str:
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="ru-RU",
        name="ru-RU-Wavenet-B"
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return base64.b64encode(response.audio_content).decode("utf-8")

# ================== ROUTES ==================

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask(data: AskRequest):
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    # ===== ЛОГИКА ОТВЕТА (ПРОСТАЯ, НО ЧЕТКАЯ) =====
    answer_text = (
        "Я вас понял. Вы спросили: "
        f"{data.question}. "
        "Я голосовой AI ассистент ARMGER GROUP. "
        "Могу отвечать на вопросы, консультировать и помогать."
    )

    audio_base64 = speak_text(answer_text)

    return {
        "text": answer_text,
        "audio": audio_base64
    }
