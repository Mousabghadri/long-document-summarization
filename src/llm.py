import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL_NAME = os.getenv("MODEL_NAME")


def generate_summary(
    prompt: str,
    model_name: str = None,
    max_tokens: int = None,
    reasoning_effort: str = None
) -> str:

    request_args = {
        "model": model_name or MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    if max_tokens is not None:
        request_args["max_tokens"] = max_tokens

    if reasoning_effort is not None:
        request_args["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(
        **request_args
    )
    choice = response.choices[0]
    content = choice.message.content
    finish_reason = choice.finish_reason

    print(
    f"[LLM] model={request_args['model']} "
    f"finish_reason={finish_reason} "
    f"max_tokens={max_tokens}"
    )

    if finish_reason == "length":
     raise RuntimeError(
        "LLM output reached the token limit. "
        "The generated text may be incomplete."
    )

    if content is None:
        raise RuntimeError(
            "The model returned no text content."
        )

    return content