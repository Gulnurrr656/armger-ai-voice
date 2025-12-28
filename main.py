import os
import json
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import texttospeech
from google.oauth2 import service_account

from openai import OpenAI

# ================== OPENAI ==================

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

# ================== SCHEMA ==================

class AskRequest(BaseModel):
    question: str

# ================== SYSTEM PROMPT ==================

SYSTEM_PROMPT = """
Ты — AI-консультант компании ARMGER GROUP (Казахстан).

О компании:
ARMGER GROUP — Казахстанская группа компаний.
Направления:
• Строительство и логистика
• IT-решения и автоматизация
• Производство медицинских расходных материалов (СИЗ)

Философия:
Практический бизнес, ответственность, прозрачность,
поддержка внутреннего производства и экономики Казахстана.

Правила ответа:
- Отвечай кратко, уверенно, профессионально
- Только по делу
- Если вопрос не по компании — вежливо верни к услугам ARMGER GROUP
- Язык: русский
"""

# ================== HELPERS ==================

def generate_answer(question: str) -> str:
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.4,
    )
    return completion.choices[0].message.content.strip()


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

    try:
        answer_text = generate_answer(data.question)
        audio_base64 = speak_text(answer_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "text": answer_text,
        "audio": audio_base64
    }
