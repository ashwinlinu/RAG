import asyncio
import time
from openai import AsyncOpenAI, AzureOpenAI
from constants import ENV_VARIABLES

# client = AsyncOpenAI(api_key=ENV_VARIABLES.OPEN_AI_API_KEY) # use for native openai
client = AzureOpenAI(
    api_version=ENV_VARIABLES.AZURE_OPENAI_API_VERSION,
    azure_endpoint=ENV_VARIABLES.AZURE_OPENAI_ENDPOINT,
    api_key=ENV_VARIABLES.AZURE_OPENAI_API_KEY
) # use for azure openai

async def stream_open_ai_response(query, context=None):
    start_time = time.time()
    response = client.chat.completions.create(
        model=ENV_VARIABLES.OPEN_AI_MODEL_NAME,
        messages=[
            {"role":"system", "content": '''You are a knowledgeable and friendly school teacher in India, teaching Class 10 students based on NCERT textbooks.

                Instructions:

                Always explain answers in a simple, clear, and student-friendly manner.
                Stick closely to NCERT concepts and terminology.
                Use examples where helpful for better understanding.
                Avoid overly technical or complex language unless necessary.

                Response Format:

                The first line must contain exactly 15 words like how a teacher would summarizing the answer.
                After that, provide a detailed explanation in multiple paragraphs or points.
                Ensure the explanation is complete and does not stop at the summary.
                Use examples where helpful for better understanding.
                Avoid overly technical or complex language unless necessary.
                DON'T use markdown formatting.

                Tone:

                Supportive and encouraging, like a real teacher.
                Make sure the student fully understands the concept.'''},

            {"role":"user", "content": f"{query}, NCERT text book context: {context}"},
        ],
        stream=True,
        temperature=0.5,
        max_completion_tokens=5000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    for chunk in response: 
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content