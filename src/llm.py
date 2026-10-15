import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL_NAME = os.getenv("MODEL_NAME")


def generate_summary(prompt: str, model_name: str = None) -> str:

    response = client.chat.completions.create(
        model=model_name or MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError(
            "The model returned no text content."
        )

    return content