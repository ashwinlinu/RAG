import os

from dotenv import load_dotenv

load_dotenv()


class ENV_VARIABLES:
    OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPEN_AI_MODEL_NAME = os.environ.get("OPENAI_GPT_MODEL_NAME")
    AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")