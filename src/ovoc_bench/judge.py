from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class JudgeConfig:
    base_url: str
    api_key: str
    model: str
    prompt_version: str
    prompt_text: str
    parallel: int = 10
    timeout_seconds: float = 120.0
    retries: int = 2


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/chat/completions"


def _extract_text_from_chat_response(payload: dict[str, Any]) -> str:
    try:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "")
    except Exception:
        return ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def build_user_prompt(prompt_template: str, *, question: str, gold_answer: str, prediction: str) -> str:
    # Use explicit placeholder replacement so literal braces in prompt templates
    # (e.g. JSON examples) are preserved instead of being interpreted by str.format.
    return (
        prompt_template
        .replace("{question}", question)
        .replace("{gold_answer}", gold_answer)
        .replace("{prediction}", prediction)
    )


async def _judge_one(
    client: httpx.AsyncClient,
    cfg: JudgeConfig,
    record: dict[str, Any],
) -> dict[str, Any]:
    url = _chat_completions_url(cfg.base_url)
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator for QA correctness. "
                    "Return only a JSON object with keys is_correct (boolean) and reasoning (string)."
                ),
            },
            {
                "role": "user",
                "content": build_user_prompt(
                    cfg.prompt_text,
                    question=str(record["question"]),
                    gold_answer=str(record["gold_answer"]),
                    prediction=str(record["prediction"]),
                ),
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    last_exc: Exception | None = None
    for attempt in range(cfg.retries + 1):
        try:
            response = await client.post(url, headers=headers, json=body, timeout=cfg.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            content = _extract_text_from_chat_response(payload)
            parsed = _parse_json_object(content) or {}
            return {
                "case_uid": record["case_uid"],
                "is_correct": bool(parsed.get("is_correct", False)),
                "reasoning": str(parsed.get("reasoning", "parse_error")),
                "judge_model_id": cfg.model,
                "judge_prompt_version": cfg.prompt_version,
                "judge_raw_json": payload,
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < cfg.retries:
                await asyncio.sleep(1.5 * (attempt + 1))
    return {
        "case_uid": record["case_uid"],
        "is_correct": False,
        "reasoning": f"judge_error: {last_exc}",
        "judge_model_id": cfg.model,
        "judge_prompt_version": cfg.prompt_version,
        "judge_raw_json": {"error": str(last_exc)},
    }


async def judge_records_async(
    records: list[dict[str, Any]],
    cfg: JudgeConfig,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(cfg.parallel)
    async with httpx.AsyncClient() as client:
        async def _wrapped(record: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await _judge_one(client, cfg, record)
        tasks = [_wrapped(record) for record in records]
        return await asyncio.gather(*tasks)


def judge_records(
    records: list[dict[str, Any]],
    cfg: JudgeConfig,
) -> list[dict[str, Any]]:
    return asyncio.run(judge_records_async(records, cfg))
