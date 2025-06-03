from typing import Annotated, Literal, Union, get_args

from fastapi import FastAPI, Query
import os


from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from apps.rag.services import qdrant_service
from apps.llm.services import openai_service

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

EmbeddingModel = Literal["BAAI/bge-m3", "paraphrase-multilingual-MiniLM-L12-v2"]


class AskParams(BaseModel):
    question: str = ''
    model_name: EmbeddingModel = 'BAAI/bge-m3'


@app.get('/ask-for-db')
async def ask_for_db(params: Annotated[AskParams, Query()]):
    results = qdrant_service.query(model_name=params.model_name, query=params.question, limit=10)
    retrieved_context = '\n'.join([hit.payload['text'] for hit in results.points])
    results = [{'score': hit.score, 'text': hit.payload['text']} for hit in results.points]
    return {'result': results}



@app.get("/ask")
async def ask(params: Annotated[AskParams, Query()]):
    results = qdrant_service.query(model_name=params.model_name, query=params.question, limit=10)
    retrieved_context = '\n'.join([hit.payload['text'] for hit in results.points])

    prompt = f"""
    다음은 사용자의 질문입니다:

    "{params.question}"

    그리고 관련된 문서들은 다음과 같습니다:

    {retrieved_context}

    이 정보를 바탕으로 질문에 친절하고 간결하게 답변해.
    """

    return StreamingResponse(openai_service.stream(prompt), media_type="text/event-stream")


@app.post('/upsert')
async def upsert_collection(texts: list[str]):
    for arg in get_args(EmbeddingModel):
        qdrant_service.upsert(model_name=arg, texts=texts)
    return {"message": "Collection upserted successfully"}


@app.get('/health')
async def health():
    return {"message": "OK"}