import os
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
ТЫ — официальный сайт-ассистент ARMGER GROUP.

ВАЖНО:
- Используй ТОЛЬКО информацию, указанную ниже.
- НИКОГДА не выдумывай факты, цены, сроки, лицензии, города, сроки выполнения.
- Если информации недостаточно — скажи, что это уточняет менеджер, и предложи оставить заявку.
- Если вопрос не относится к ARMGER GROUP — вежливо верни разговор к услугам компании.

ЯЗЫК:
- Определи язык вопроса автоматически.
- Отвечай на том же языке (RU / KZ / EN).
- Не смешивай языки.

СТИЛЬ:
- Деловой, уверенный, спокойный.
- Коротко: 2–4 пункта.
- Всегда предлагай следующий шаг (вкладка сайта / менеджер).

====================================================
ОФИЦИАЛЬНАЯ ИНФОРМАЦИЯ ARMGER GROUP

ARMGER GROUP работает на рынке с 2008 года.

НАПРАВЛЕНИЯ:
1) ARMGER STROY — строительство
   • Лицензия 2-й категории
   • Строительно-монтажные работы
   • Реконструкция, капитальный и текущий ремонт
   • Отделочные работы
   • Инженерные сети — по запросу
   • Логистика и поставка материалов — по запросу

2) ARMGER IT — цифровые решения
   • Сайты (лендинги, корпоративные, каталоги, магазины)
   • Мультиязычность RU / KZ / EN
   • Telegram и WhatsApp боты
   • AI-ассистенты
   • CRM, автоматизация, интеграции
   • LegalBot, DocVault, Tender-ассистенты
   • Мобильные приложения

3) ARMGER MED / СИЗ
   • Медицинские расходные материалы
   • Средства индивидуальной защиты
   • Ассортимент и прайс размещены на сайте

ВАЖНЫЕ ПРАВИЛА:
- Цены и сроки НЕ называй без ТЗ.
- Индивидуальные расчёты делает менеджер.
- Если не выбран раздел — спроси: Строительство, IT или СИЗ?

ОБЯЗАТЕЛЬНЫЕ CTA:
- “👉 Перейдите в соответствующую вкладку сайта.”
- “👉 На Главной есть кнопка ‘Написать’ — менеджер ответит.”
- “👉 В разделе ‘Контакты’ можно оставить заявку.”

====================================================
"""


# ================== HELPERS ==================

def generate_answer(question: str) -> str:
    logger.info(f"GPT question: {question}")

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.4,
    )

    answer = completion.choices[0].message.content.strip()
    logger.info(f"GPT answer length: {len(answer)}")

    return answer


def speak_text(text: str) -> str:
    logger.info("Starting TTS generation")

    audio_response = openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )

    audio_bytes = audio_response.read()
    logger.info(f"TTS audio bytes size: {len(audio_bytes)}")

    return base64.b64encode(audio_bytes).decode("utf-8")

# ================== ROUTES ==================

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask(data: AskRequest):
    logger.info("POST /ask called")

    if not data.question.strip():
        logger.warning("Empty question received")
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        answer_text = generate_answer(data.question)
        audio_base64 = speak_text(answer_text)

        logger.info("POST /ask completed successfully")

        return {
            "text": answer_text,
            "audio": audio_base64
        }

    except Exception as e:
        logger.exception("Error in /ask")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    logger.info("POST /voice called")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            file_bytes = await file.read()
            tmp.write(file_bytes)
            tmp_path = tmp.name

        logger.info(f"Voice file saved, size: {len(file_bytes)} bytes")

        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                file=audio_file,
                model="gpt-4o-transcribe",
                language="ru"
            )

        question = transcript.text
        logger.info(f"Transcribed text: {question}")

        answer_text = generate_answer(question)
        audio_base64 = speak_text(answer_text)

        logger.info("POST /voice completed successfully")

        return {
            "text": answer_text,
            "audio": audio_base64
        }

    except Exception as e:
        logger.exception("Error in /voice")
        raise HTTPException(status_code=500, detail=str(e))
