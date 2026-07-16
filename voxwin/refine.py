"""Optional post-transcription refinement through a local Ollama server.

Mirrors VoxLocal's Ollama integration. By default the endpoint is
127.0.0.1 (the URL is user-configurable — pointing it elsewhere sends
dictated text to that host, so keep it local). Failure of the refinement
step never blocks dictation — the caller falls back to unrefined text.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import urllib.error
import urllib.request

from .config import Config

log = logging.getLogger(__name__)


class RefinementError(Exception):
    pass


def refine_text(text: str, cfg: Config) -> str:
    prompt_template = cfg.ollama_prompt or "{text}"
    if "{text}" in prompt_template:
        prompt = prompt_template.replace("{text}", text)
    else:
        prompt = f"{prompt_template}\n\n{text}"

    payload = json.dumps(
        {
            "model": cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")

    url = cfg.ollama_url.rstrip("/") + "/api/generate"
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    # Ollama is local — never route it through a system proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=cfg.ollama_timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        http.client.HTTPException,  # e.g. IncompleteRead on a dying Ollama
        OSError,
        ValueError,
    ) as exc:
        raise RefinementError(f"Ollama недоступна ({exc})") from exc

    refined = (body.get("response") or "").strip()
    if not refined:
        raise RefinementError("Ollama вернула пустой ответ")

    # Models love wrapping answers in quotes/code fences — strip them.
    # Only the opening fence line may carry a language tag; touching anything
    # after the first newline would eat words like "Text me when..." from
    # the actual content.
    if refined.startswith("```"):
        refined = re.sub(r"^```[\w-]*[ \t]*\n?", "", refined)
        refined = re.sub(r"\n?```\s*$", "", refined).strip()
    if len(refined) >= 2 and refined[0] in "\"«'" and refined[-1] in "\"»'":
        refined = refined[1:-1].strip()

    # Paranoia guard: if the model replied with something wildly different
    # in size, it's probably chatter, not a correction.
    if refined and 0.3 <= len(refined) / max(len(text), 1) <= 3.0:
        return refined
    raise RefinementError("Ответ Ollama не похож на исправленный текст")
