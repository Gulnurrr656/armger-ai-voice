import os
import base64
import tempfile
import logging
import re

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ================== LOGGING ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ================== OPENAI ==================

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
logger.info("OpenAI client initialized")

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
Ты — официальный AI-ассистент компании ARMGER GROUP (Казахстан).

СТРОГО:
- Используй ТОЛЬКО информацию ниже.
- НИКОГДА не выдумывай факты, цены, сроки, лицензии.
- Если информации нет — скажи, что уточнит менеджер.
- Если вопрос не про ARMGER GROUP — верни разговор к услугам компании.

ЯЗЫК:
- Определи язык вопроса.
- Отвечай на том же языке (RU / KZ / EN).
- Языки не смешивай.

СТИЛЬ:
- Деловой
- Коротко (2–4 пункта)
- В конце CTA (вкладка сайта / менеджер)
"""

# ================== HELPERS ==================

def detect_lang(text: str) -> str:
    text = text.lower()

    # Казахский
    if re.search(r"[әғқңөұүі]", text):
        return "kk"

    # Русский — кириллица
    if re.search(r"[а-яё]", text):
        return "ru"

    # Английский
    return "en"


def select_voice(lang: str) -> str:
    # Лучшее качество
    if lang in ("ru", "kk"):
        return "nova"
    return "verse"


def generate_answer(question: str) -> str:
    logger.info(f"GPT question: {question}")

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
    )

    answer = completion.choices[0].message.content.strip()
    logger.info(f"GPT answer length: {len(answer)}")
    return answer


def speak_text(text: str) -> str:
    lang = detect_lang(text)
    voice = select_voice(lang)

    logger.info(f"TTS lang={lang}, voice={voice}")

    audio_response = openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )

    audio_bytes = audio_response.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

# ================== ROUTES ==================

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask(data: AskRequest):
    logger.info("POST /ask")

    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        answer_text = generate_answer(data.question)
        audio_base64 = speak_text(answer_text)

        return {
            "text": answer_text,
            "audio": audio_base64
        }

    except Exception:
        logger.exception("ASK ERROR")
        raise HTTPException(status_code=500, detail="AI processing error")


@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    logger.info("POST /voice")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            audio_bytes = await file.read()
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                file=audio_file,
                model="gpt-4o-transcribe",
                language="ru"
            )

        question = transcript.text
        logger.info(f"Voice text: {question}")

        answer_text = generate_answer(question)
        audio_base64 = speak_text(answer_text)

        return {
            "text": answer_text,
            "audio": audio_base64
        }

    except Exception:
        logger.exception("VOICE ERROR")
        raise HTTPException(status_code=500, detail="Voice processing error")
