from dotenv import load_dotenv
import os
import openai
from typing import Any, Iterable, cast

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


