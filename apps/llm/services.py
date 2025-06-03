import os
from openai import AsyncOpenAI


class OpenAIService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.default_model_name
        self.openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @property
    def default_model_name(self):
        return "gpt-4o-mini"

    async def stream(self, prompt: str):
        stream = await self.openai_client.chat.completions.create(model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True)

        async for chunk in stream:
            if chunk.choices[0].finish_reason == "stop":
                break
            yield chunk.choices[0].delta.content


openai_service = OpenAIService()