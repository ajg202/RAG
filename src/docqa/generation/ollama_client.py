import re
import warnings
from types import SimpleNamespace

import requests

from docqa.config import settings

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

#_parse_think interprets the think option for Ollama. It returns True, False, or None (to omit the option) based on the input string.
def _parse_think(value: str) -> bool | None:
    value = value.strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return None  #don't send the option at all

#OllamaClient is a stand-in for the anthropic.Anthropic client, providing a similar interface for sending messages to an Ollama model. It constructs the request payload, handles errors, and returns a simplified response object.
class OllamaClient:
    #_init_ initializes the OllamaClient with configuration options such as the server URL, context size, temperature, think option, and timeout. It sets up the messages attribute to allow calling create(...) directly.
    def __init__(self, url=None, num_ctx=None, temperature=None, think="auto", timeout=300):
        self._url = (url or settings.ollama_url).rstrip("/")
        self._num_ctx = num_ctx or settings.ollama_num_ctx
        self._temperature = settings.ollama_temperature if temperature is None else temperature
        self._think = _parse_think(settings.ollama_think) if think == "auto" else think
        self._timeout = timeout  #the first request loads the model into memory
        self.messages = self  #so client.messages.create(...) works

    #create sends a message to the Ollama model with the specified parameters. It constructs the request payload, makes the HTTP POST request, handles errors, and returns a simplified response object containing the generated text and stop reason.
    def create(self, *, model, max_tokens, system, messages):
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": {
                "num_ctx": self._num_ctx,
                "num_predict": max_tokens,
                "temperature": self._temperature,
            },
        }
        if self._think is not None:
            payload["think"] = self._think

        try:
            resp = requests.post(f"{self._url}/api/chat", json=payload, timeout=self._timeout)
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Can't reach Ollama at {self._url}. Is it running? (try: ollama serve)"
            ) from e
        except requests.Timeout as e:
            raise RuntimeError(f"Ollama didn't answer within {self._timeout} seconds.") from e

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            if resp.status_code == 404:
                raise RuntimeError(f"Ollama doesn't have the model {model!r}. Run: ollama pull {model}")
            raise RuntimeError(f"Ollama returned {resp.status_code}: {detail}")

        data = resp.json()

        used = data.get("prompt_eval_count")
        if used is not None and used >= self._num_ctx * 0.98:
            warnings.warn(
                f"The prompt used {used} of {self._num_ctx} context tokens, so Ollama may have "
                "silently dropped part of it. Raise OLLAMA_NUM_CTX or lower TOP_K."
            )

        text = _THINK_BLOCK.sub("", data.get("message", {}).get("content", "")).strip()
        stop = "max_tokens" if data.get("done_reason") == "length" else "end_turn"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=stop,
        )