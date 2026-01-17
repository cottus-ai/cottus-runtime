import asyncio
import time
import json
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import uuid

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None
    BaseModel = object
    StreamingResponse = None
    JSONResponse = None
    HTTPException = None
    Request = None
    Field = None

from .async_engine import AsyncEngine


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    top_k: Optional[int] = 50
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class CottusServer:
    def __init__(self, engine: AsyncEngine, tokenizer: Any):
        if FastAPI is None:
            raise RuntimeError("FastAPI not installed")
        self._engine = engine
        self._tokenizer = tokenizer
        self._app = FastAPI(title="Cottus API", version="0.2.0")
        self._setup_routes()

    @property
    def app(self):
        return self._app

    def _setup_routes(self):
        @self._app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            return await self._handle_chat_completions(request)

        @self._app.get("/v1/models")
        async def list_models():
            return {"object": "list", "data": [{"id": "cottus-default", "object": "model", "created": int(time.time()), "owned_by": "cottus"}]}

        @self._app.get("/health")
        async def health():
            return {"status": "ok"}

    async def _handle_chat_completions(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, StreamingResponse]:
        prompt = self._build_prompt(request.messages)
        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        request_id = "chatcmpl-" + uuid.uuid4().hex[:8]
        created = int(time.time())

        if request.stream:
            return StreamingResponse(self._stream_response(request_id, created, request.model, prompt_ids, request.max_tokens or 256, request.temperature or 0.0, request.top_k or 50), media_type="text/event-stream")

        output_ids = await self._engine.generate(prompt_ids=prompt_ids, max_tokens=request.max_tokens or 256, temperature=request.temperature or 0.0, top_k=request.top_k or 50)
        output_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)

        return ChatCompletionResponse(id=request_id, created=created, model=request.model, choices=[ChatCompletionChoice(index=0, message=ChatMessage(role=MessageRole.ASSISTANT, content=output_text), finish_reason="stop")], usage=ChatCompletionUsage(prompt_tokens=len(prompt_ids), completion_tokens=len(output_ids), total_tokens=len(prompt_ids) + len(output_ids)))

    async def _stream_response(self, request_id: str, created: int, model: str, prompt_ids: List[int], max_tokens: int, temperature: float, top_k: int) -> AsyncGenerator[str, None]:
        first_chunk = ChatCompletionChunk(id=request_id, created=created, model=model, choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(role="assistant"), finish_reason=None)])
        yield "data: " + first_chunk.model_dump_json() + "\n\n"

        async for token_id in self._engine.generate_stream(prompt_ids=prompt_ids, max_tokens=max_tokens, temperature=temperature, top_k=top_k):
            token_text = self._tokenizer.decode([token_id], skip_special_tokens=True)
            chunk = ChatCompletionChunk(id=request_id, created=created, model=model, choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(content=token_text), finish_reason=None)])
            yield "data: " + chunk.model_dump_json() + "\n\n"

        final_chunk = ChatCompletionChunk(id=request_id, created=created, model=model, choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(), finish_reason="stop")])
        yield "data: " + final_chunk.model_dump_json() + "\n\n"
        yield "data: [DONE]\n\n"

    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        parts = []
        sys_tag = "<" + "|system|" + ">"
        user_tag = "<" + "|user|" + ">"
        asst_tag = "<" + "|assistant|" + ">"
        end_tag = "<" + "/s" + ">"
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                parts.append(sys_tag + "\n" + msg.content + end_tag)
            elif msg.role == MessageRole.USER:
                parts.append(user_tag + "\n" + msg.content + end_tag)
            elif msg.role == MessageRole.ASSISTANT:
                parts.append(asst_tag + "\n" + msg.content + end_tag)
        parts.append(asst_tag + "\n")
        return "".join(parts)
