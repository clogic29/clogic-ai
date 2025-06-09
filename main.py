from typing import Annotated, Literal, Union, get_args

from fastapi import FastAPI, Query, Form, BackgroundTasks
import os


from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from apps.rag.services import qdrant_service
from apps.llm.services import openai_service
from slack_sdk.web.async_client import AsyncWebClient

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

# Slack 클라이언트 초기화
slack_client = AsyncWebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

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


async def process_slack_interaction(channel_id: str, command_text: str):
    """백그라운드에서 전체 Slack 상호작용을 처리"""
    try:
        # 질문을 채널에 표시
        initial_response = await slack_client.chat_postMessage(
            channel=channel_id,
            text=f"질문: {command_text}"
        )
        
        # 스레드로 로딩 메시지 전송
        loading_response = await slack_client.chat_postMessage(
            channel=channel_id,
            thread_ts=initial_response["ts"],
            text="벡터 DB에 질의 중..."
        )
        
        # AI 응답 생성
        results = qdrant_service.query(model_name='BAAI/bge-m3', query=command_text, limit=3)
        retrieved_context = '\n'.join([hit.payload['text'] for hit in results.points])

        await slack_client.chat_postMessage(
            channel=channel_id,
            thread_ts=initial_response["ts"],
            text=f"벡터 DB 질의 결과 : \n```{retrieved_context}```"
        )
        

        await slack_client.chat_update(
            channel=channel_id,
            ts=loading_response["ts"],
            text='LLM에 질의 중...'
        )
        
        if retrieved_context:
            prompt = f"""
            다음은 사용자의 질문입니다:

            "{command_text}"

            그리고 관련된 문서들은 다음과 같습니다:

            {retrieved_context}

            이 정보를 바탕으로 질문에 친절하고 간결하게 답변해주세요.
            """
            
            # 스트림을 완전히 소비하여 응답 텍스트 생성
            response_chunks = []
            async for chunk in openai_service.stream(prompt):
                if chunk:
                    response_chunks.append(chunk)
            
            response_text = ''.join(response_chunks)
        else:
            response_text = f"'{command_text}'에 대한 정보를 찾을 수 없습니다."
        
        # 로딩 메시지를 실제 AI 응답으로 업데이트
        await slack_client.chat_update(
            channel=channel_id,
            ts=initial_response["ts"],
            text=response_text
        )

        await slack_client.chat_update(
            channel=channel_id,
            ts=loading_response["ts"],
            text='답변 완료'
        )
        
    except Exception as e:
        # 에러 발생 시 에러 메시지 전송
        await slack_client.chat_postMessage(
            channel=channel_id,
            text=f"응답 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.post('/slack/ask')
async def slack_command(
    background_tasks: BackgroundTasks,
    token: Annotated[str, Form()],
    team_id: Annotated[str, Form()],
    team_domain: Annotated[str, Form()],
    channel_id: Annotated[str, Form()],
    channel_name: Annotated[str, Form()],
    user_id: Annotated[str, Form()],
    user_name: Annotated[str, Form()],
    command: Annotated[str, Form()],
    text: Annotated[str, Form()] = '',
    response_url: Annotated[str, Form()] = ''
):
    """
    Slack slash command를 처리하는 엔드포인트
    """
    command_text = text.strip()
    
    if not command_text:
        return {
            "response_type": "ephemeral",
            "text": "안녕하세요! 무엇을 도와드릴까요?"
        }
    
    # 모든 처리를 백그라운드로 이동
    background_tasks.add_task(
        process_slack_interaction,
        channel_id,
        command_text
    )
    
    # 즉시 200 응답 반환 (Slack에서 요구하는 3초 이내 응답)
    return {"response_type": "ephemeral", "text": "질문을 처리하고 있습니다."}


@app.post('/slack/append')
async def slack_append_command(
    token: Annotated[str, Form()],
    team_id: Annotated[str, Form()],
    team_domain: Annotated[str, Form()],
    channel_id: Annotated[str, Form()],
    channel_name: Annotated[str, Form()],
    user_id: Annotated[str, Form()],
    user_name: Annotated[str, Form()],
    command: Annotated[str, Form()],
    text: Annotated[str, Form()] = '',
    response_url: Annotated[str, Form()] = '',
    trigger_id: Annotated[str, Form()] = ''
):
    """
    텍스트 추가를 위한 Slack slash command 처리
    """
    try:
        # 모달 창 열기
        modal_view = {
            "type": "modal",
            "callback_id": "append_text_modal",
            "title": {
                "type": "plain_text",
                "text": "텍스트 추가"
            },
            "submit": {
                "type": "plain_text",
                "text": "추가"
            },
            "close": {
                "type": "plain_text",
                "text": "취소"
            },
            "blocks": [
                {
                    "type": "input",
                    "block_id": "text_input_block",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "text_input",
                        "multiline": True,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "추가할 텍스트를 입력하세요..."
                        }
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "텍스트 내용"
                    }
                }
            ]
        }
        
        await slack_client.views_open(
            trigger_id=trigger_id,
            view=modal_view
        )
        
        return {"ok": True}
        
    except Exception as e:
        return {
            "response_type": "ephemeral",
            "text": f"모달을 여는 중 오류가 발생했습니다: {str(e)}"
        }


async def process_text_upsert(input_text: str, user_id: str):
    """백그라운드에서 텍스트 업로드 처리"""
    try:
        for model_name in get_args(EmbeddingModel):
            qdrant_service.upsert(model_name=model_name, texts=[input_text])
        
        # 성공 메시지를 사용자에게 DM으로 전송
        await slack_client.chat_postMessage(
            channel=user_id,
            text=f"텍스트가 성공적으로 추가되었습니다:\n```{input_text[:100]}{'...' if len(input_text) > 100 else ''}```"
        )
    except Exception as upsert_error:
        print(f"Upsert 오류: {upsert_error}")
        # 에러 발생 시 사용자에게 DM으로 알림
        await slack_client.chat_postMessage(
            channel=user_id,
            text=f"텍스트 추가 중 오류가 발생했습니다: {str(upsert_error)}"
        )

@app.post('/slack/modal')
async def slack_modal_submission(
    payload: Annotated[str, Form()]
):
    """
    Slack 모달에서 제출된 데이터 처리
    """
    import json
    import asyncio
    
    # 즉시 성공 응답 반환 (모달 닫기) - 어떤 검증도 하지 않음
    
    # 모든 처리를 완전히 분리된 태스크로 실행
    asyncio.create_task(handle_modal_data(payload))
    
    return {}

async def handle_modal_data(payload: str):
    """모달 데이터를 완전히 분리해서 처리"""
    import json
    
    try:
        # payload는 JSON 문자열로 전송됨
        data = json.loads(payload)
        
        # 제출 타입 확인
        submission_type = data.get("type")
        if submission_type != "view_submission":
            return
        
        # 모달 callback_id 확인
        if data.get("view", {}).get("callback_id") != "append_text_modal":
            return
        
        # 입력된 텍스트 추출
        values = data.get("view", {}).get("state", {}).get("values", {})
        text_input_block = values.get("text_input_block", {})
        text_input = text_input_block.get("text_input", {})
        input_text = text_input.get("value", "").strip()
        
        # 사용자 ID 추출
        user_id = data.get("user", {}).get("id", "unknown")
        
        if not input_text:
            # 빈 텍스트인 경우 사용자에게 알림
            await slack_client.chat_postMessage(
                channel=user_id,
                text="텍스트가 입력되지 않았습니다."
            )
            return
        
        # 텍스트 업로드 처리
        await process_text_upsert(input_text, user_id)
        
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"Payload: {payload}")
    except Exception as e:
        print(f"모달 처리 오류: {e}")
        print(f"Payload: {payload}")

@app.get('/health')
async def health():
    return {"message": "OK"}


