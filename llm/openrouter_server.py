import os
import logging
import requests
import json
import base64
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api")
API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "YOUR_OPENROUTER_API_KEY"
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load credentials
if not API_KEY:
    logger.warning("OPENROUTER_API_KEY is not set.")

logger.info(f"Using OpenRouter base URL: {API_BASE}")

with open("config.json", encoding="utf-8") as f:
    config = json.load(f)

PROMPT = config.get("PROMPT")
SHOT_PROMPT = config.get("SHOT_PROMPT")

# Pydantic models
class ChatHistoryItem(BaseModel):
    role: str
    content: str

class TemplateData(BaseModel):
    templateDescription: str

class AnswerRequestDto(BaseModel):
    text: str
    templateData: Optional[TemplateData] = None

    # в автоматическом режиме ответов на вопросы приходит manualRequest = false, при ручном запросе ответа = true.
    # нужно вернуть строку "NOQ" в потоке ответа, если отвечать не на что (вопрос не обнаружен)
    # на данный момент этот функционал не реализован в данном скрипте, но вы можете написать собственную реализацию.
    manualRequest: bool = Field(False, alias="manualRequest")

    # устанавливется в true когда ответ запрашивается повторно при правом клике на сообщении в чате
    retryQuestion: bool = Field(False, alias="retryQuestion")

    # устанавливется в true когда запрашивается думающий шот
    shotThinkingModel: bool = Field(False, alias="shotThinkingModel")

    chatHistory: List[ChatHistoryItem] = []

def _stream_from_openrouter(messages, model):
    url = f"{API_BASE}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }

    def event_generator():
        buffer = ""
        # в данном случае проверка сертификата отключена, если надо, можно включить => verify=True
        with requests.post(url, headers=headers, json=payload, stream=True, verify=False) as r:
            if r.status_code != 200:
                body = r.text
                logger.error(f"OpenRouter API returned {r.status_code}: {body}")
                raise HTTPException(status_code=500, detail="OpenRouter API error")
            for chunk in r.iter_content(chunk_size=1024):
                chunk = chunk.decode('utf-8', errors='replace')  # Декодируем с заменой некорректных символов
                buffer += chunk
                while True:
                    line_end = buffer.find('\n')
                    if line_end == -1:
                        break
                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end+1:]
                    if not line.startswith('data: '):
                        continue
                    data = line[len('data: '):]
                    if data == '[DONE]':
                        return
                    try:
                        obj = json.loads(data)
                        delta = obj.get('choices', [])[0].get('delta', {})
                        content = delta.get('content')
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    return StreamingResponse(event_generator(), media_type="text/event-stream; charset=utf-8")

@app.post("/answer/stream", response_class=StreamingResponse)
def answer_stream(request: Request, dto: AnswerRequestDto):
    topic = dto.templateData.templateDescription if dto.templateData else dto.templateDescription
    system_prompt = PROMPT.replace("%TEMPLATE%", topic or "")

    messages = [{"role": "system", "content": system_prompt}]
    for msg in dto.chatHistory:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": dto.text})

    model = config.get("ANSWER_MODEL")

    return _stream_from_openrouter(messages, model)

@app.post("/answer/shot/stream", response_class=StreamingResponse)
def get_answer_stream_by_screenshot_with_image(
    requestDto: str = Form(...),
    screenshot: UploadFile = File(...)
):
    dto = AnswerRequestDto.parse_raw(requestDto)
    image_bytes = screenshot.file.read()
    image_url = f"data:{screenshot.content_type};base64,{base64.b64encode(image_bytes).decode()}"

    topic = dto.templateData.templateDescription if dto.templateData else dto.templateDescription
    system_prompt = SHOT_PROMPT.replace("%TEMPLATE%", topic or "")

    messages = [{"role": "system", "content": system_prompt}]
    for msg in dto.chatHistory:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": dto.text},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    })

    if dto.shotThinkingModel:
        model = config.get("THINKING_SHOT_MODEL")
    else:
        model = config.get("SHOT_MODEL")

    return _stream_from_openrouter(messages, model)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("openrouter_server:app", host="0.0.0.0", port=8000, reload=True, log_config=None)