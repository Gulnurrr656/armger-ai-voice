import os
import re
import base64
import tempfile
import logging

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

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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

# ================== LANGUAGE ==================

def detect_lang(text: str) -> str:
    t = text.lower()
    if re.search(r"[әғқңөұүі]", t):
        return "kk"
    if re.search(r"[a-z]", t):
        return "en"
    return "ru"

def select_voice(lang: str) -> str:
    if lang in ("ru", "kk"):
        return "nova"
    return "verse"

# ================== PROMPTS ==================

SYSTEM_PROMPTS = {
    "ru": """ТЫ — ARMGER GROUP Assistant (сайт-ассистент ARMGER GROUP).

Ты встроен прямо в официальный сайт ARMGER GROUP.
НЕ говори «перейдите на сайт». Используй формулировки:
«на этой странице», «в соответствующем разделе», «в меню сайта».

Ты консультируешь по направлениям:
ARMGER STROY, ARMGER IT, ARMGER MED / СИЗ.

Используй ТОЛЬКО информацию ниже.
НИКОГДА не выдумывай цены, сроки, лицензии.
Если данных недостаточно — скажи, что менеджер уточнит.

Отвечай ТОЛЬКО на русском языке.
Коротко, делово, 2–4 пункта + следующий шаг.

=== ДАЛЕЕ ПОЛНЫЙ ПРОМПТ ИЗ ФАЙЛА (RU) ===
""",

    "en": """You are the ARMGER GROUP website assistant.

You are embedded directly on the official ARMGER GROUP website.
Do NOT say “visit the website”. Use:
“on this page”, “in the relevant section”, “in the site menu”.

Use ONLY the information provided below.
Do NOT invent prices, timelines, licenses.
If data is missing — say the manager will clarify.

Answer STRICTLY in English.
Business tone, 2–4 bullet points + next step.

=== FULL EN PROMPT CONTENT BELOW ===
""",

    "kk": """СЕН — ARMGER GROUP сайтының ассистентісің.

Сен сайттың ІШІНДЕ жұмыс істейсің.
«Сайтқа өтіңіз» деп айтпа.
«Осы бетте», «тиісті бөлімде», «сайт мәзірінде» деп айт.

Тек төмендегі ақпаратты қолдан.
Баға, мерзім, лицензия ойдан қоспа.
Қажет болса — менеджер нақтылайды де.

ЖАУАПТЫ ТЕК ҚАЗАҚ ТІЛІНДЕ бер.
Қысқа, іскер стиль, 2–4 тармақ + келесі қадам.

=== ТОЛЫҚ ҚАЗАҚША ПРОМПТ ТӨМЕНДЕ ===
"""
}

# ВАЖНО: сюда ты можешь просто вставить
# ПОЛНЫЙ ТЕКСТ ИЗ DOCX без изменений
# (я оставил маркеры, логика уже готова)

# ================== HELPERS ==================

def generate_answer(question: str) -> tuple[str, str]:
    lang = detect_lang(question)
    system_prompt = SYSTEM_PROMPTS[lang]

    logger.info(f"Language detected: {lang}")
    logger.info(f"User question: {question}")

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
    )

    answer = completion.choices[0].message.content.strip()
    logger.info(f"Answer length: {len(answer)}")

    return answer, lang

def speak_text(text: str, lang: str) -> str:
    voice = select_voice(lang)
    logger.info(f"TTS voice: {voice} | lang: {lang}")

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )

    audio_bytes = response.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

# ================== ROUTES ==================

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask(data: AskRequest):
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        answer, lang = generate_answer(data.question)
        audio = speak_text(answer, lang)
        return {"text": answer, "audio": audio}
    except Exception as e:
        logger.exception("ASK ERROR")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f
            )

        question = transcript.text
        answer, lang = generate_answer(question)
        audio = speak_text(answer, lang)

        return {"text": answer, "audio": audio}

    except Exception as e:
        logger.exception("VOICE ERROR")
        raise HTTPException(status_code=500, detail=str(e))
