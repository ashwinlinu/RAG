import asyncio
from openai import AsyncOpenAI
from constants import ENV_VARIABLES

client = AsyncOpenAI(api_key=ENV_VARIABLES.OPEN_AI_API_KEY)


async def stream_open_ai_response(query):
    stream = await client.chat.completions.create(
        model=ENV_VARIABLES.OPEN_AI_MODEL_NAME,
        messages=[
            {"role":"system", "content": "you are an human teacher in india who teaches 10th class, so answer the students questions accordingly,"
            "note the first chunk should always contain 5 words only and rest info in rest of the chunks"},
            {"role":"user", "content": f"{query}"}
        ],
        stream=True
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content