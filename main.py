import os
import base64
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI

# ================== OPENAI ==================

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
ARMGER GROUP — Казахстанская группа компаний.
Направления:
• Строительство и логистика
• IT-решения и автоматизация
• Производство медицинских расходных материалов (СИЗ)

Философия:
Практический бизнес, ответственность, прозрачность,
поддержка внутреннего производства и экономики Казахстана.

Правила ответа:
- Отвечай кратко, уверенно, профессионально
- Только по делу
- Если вопрос не по компании — вежливо верни к услугам ARMGER GROUP
- Язык: русский
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
    audio = openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",  # современный, живой голос
        input=text
    )
    return base64.b64encode(audio).decode("utf-8")

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


@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    try:
        # сохраняем голос во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Speech-to-Text
        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                file=audio_file,
                model="gpt-4o-transcribe",
                language="ru"
            )

        question = transcript.text

        # GPT + TTS
        answer_text = generate_answer(question)
        audio_base64 = speak_text(answer_text)

        return {
            "text": answer_text,
            "audio": audio_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
