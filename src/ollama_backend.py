"""Local Ollama backend. Email text never leaves the machine."""
from __future__ import annotations

import time


class OllamaBackend:
    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        *,
        max_retries: int = 30,
        retry_wait: float = 10.0,
    ):
        # Imported lazily so a clear error is raised if the package is missing.
        from ollama import Client

        self.model = model
        self._client = Client(host=host)
        # Overnight resilience: if Ollama briefly goes away (restart, app reload,
        # machine wakes from sleep), wait and retry instead of crashing the whole
        # run. max_retries * retry_wait is the longest we'll wait for it to return
        # (default ~5 minutes) before giving up on a single email.
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def chat(
        self,
        system: str,
        prompt: str,
        *,
        json: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        attempt = 0
        while True:
            try:
                resp = self._client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    format="json" if json else "",
                    options={"num_predict": max_tokens, "temperature": temperature},
                )
                return resp["message"]["content"]
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(
                        f"Ollama request failed for model '{self.model}' after "
                        f"{self.max_retries} retries. Is Ollama running ('ollama list') "
                        f"and the model pulled ('ollama pull {self.model}')?\n"
                        f"Underlying error: {e}"
                    ) from e
                print(
                    f"    ! Ollama unavailable ({e.__class__.__name__}); retry "
                    f"{attempt}/{self.max_retries} in {self.retry_wait:.0f}s. "
                    f"Start Ollama and this continues on its own.",
                    flush=True,
                )
                time.sleep(self.retry_wait)
