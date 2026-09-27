from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def api_configured(repo_root: Path) -> bool:
    load_dotenv(repo_root / ".env", override=False)
    return bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))


def summarize_results(repo_root: Path, context: dict) -> str:
    if not api_configured(repo_root):
        raise ValueError("OPENAI_API_KEY and OPENAI_MODEL must be set in .env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.responses.create(
        model=os.environ["OPENAI_MODEL"],
        store=False,
        input=[
            {
                "role": "system",
                "content": (
                    "Explain the supplied retail forecast metrics to an analyst in at most "
                    "five short sentences. Use only supplied values; do not invent figures "
                    "or claim measured inventory savings, causal effects, or production "
                    "impact. Mention one meaningful limitation. The forecasts and metrics "
                    "are computed elsewhere and cannot be changed by this response."
                ),
            },
            {"role": "user", "content": json.dumps(context, default=str)},
        ],
    )
    return response.output_text
