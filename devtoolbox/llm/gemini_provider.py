"""Google Gemini provider using native SDK (google-genai) only."""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import os

from google import genai as google_genai
from google.genai import types as genai_types

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from devtoolbox.llm.provider import (
    BaseLLMConfig,
    BaseLLMProvider,
    register_provider,
    register_config,
)

logger = logging.getLogger(__name__)


@register_config('gemini')
@dataclass
class GeminiConfig(BaseLLMConfig):
    """Gemini config; loads from env when not provided."""

    api_key: str = field(
        default_factory=lambda: (
            os.environ.get('GOOGLE_API_KEY') or
            os.environ.get('GEMINI_API_KEY')
        ),
    )
    model: str = field(
        default_factory=lambda: os.environ.get(
            'GEMINI_MODEL',
            'gemini-2.5-flash-lite',
        ),
    )
    temperature: float = field(
        default_factory=lambda: float(
            os.environ.get('GEMINI_TEMPERATURE', '0.7'),
        ),
    )
    max_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get('GEMINI_MAX_TOKENS', '80000'),
        ),
    )
    top_p: float = field(
        default_factory=lambda: float(
            os.environ.get('GEMINI_TOP_P', '1.0'),
        ),
    )
    top_k: int = field(
        default_factory=lambda: int(
            os.environ.get('GEMINI_TOP_K', '40'),
        ),
    )

    def __post_init__(self):
        self._log_config()
        self._validate()

    def _log_config(self):
        if not self.api_key and not (
            os.environ.get('GOOGLE_API_KEY') or
            os.environ.get('GEMINI_API_KEY')
        ):
            logger.error(
                "Google Gemini API key not found in constructor or "
                "environment",
            )
        logger.debug(
            f"Gemini config: model={self.model}, "
            f"temperature={self.temperature}",
        )

    def _validate(self):
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key is required. Set via constructor "
                "or GOOGLE_API_KEY/GEMINI_API_KEY env.",
            )

    @classmethod
    def from_env(cls) -> 'GeminiConfig':
        """Backward-compat: create config from env."""
        logger.warning(
            "from_env() deprecated; config loads from env.",
        )
        return cls()


class GeminiError(Exception):
    """Gemini API errors."""
    pass


class GeminiRateLimitError(GeminiError):
    """Gemini rate limit exceeded."""
    pass


def _extract_text(resp: Any) -> str:
    """Get text from generate_content response."""
    if hasattr(resp, "text"):
        return resp.text or ""
    if hasattr(resp, "candidates") and resp.candidates:
        c = resp.candidates[0]
        if hasattr(c, "content") and c.content and c.content.parts:
            return c.content.parts[0].text or ""
    return str(resp)


@register_provider('GeminiProvider')
class GeminiProvider(BaseLLMProvider):
    """Gemini provider via google-genai only."""

    def __init__(self, config: GeminiConfig):
        logger.debug(
            f"Gemini provider: model={config.model}, "
            f"temperature={config.temperature}",
        )
        super().__init__(config)
        self.config = config
        self._client = google_genai.Client(api_key=config.api_key)

    def _to_contents(
        self,
        messages: List[Dict[str, str]],
    ) -> Union[str, List[Any]]:
        """Convert messages to SDK contents (str or list of Content)."""
        if not messages:
            return ""
        if len(messages) == 1 and messages[0].get("role") == "user":
            return messages[0].get("content", "")

        out = []
        for m in messages:
            role = "model" if m.get("role") == "assistant" else "user"
            text = m.get("content", "")
            if m.get("role") not in ("user", "assistant", "system"):
                logger.warning(
                    f"Unknown role {m.get('role')}, treating as user",
                )
            out.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=text)],
                ),
            )
        return out

    def _config(
        self,
        max_tokens: Optional[int],
        temperature: Optional[float],
        json_mode: bool,
        response_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build generate_content config dict."""
        cfg: Dict[str, Any] = {}
        t = (
            temperature
            if temperature is not None
            else self.config.temperature
        )
        if t is not None:
            cfg["temperature"] = t
        n = (
            max_tokens
            if max_tokens is not None
            else self.config.max_tokens
        )
        if n is not None:
            cfg["max_output_tokens"] = n
        if self.config.top_p is not None:
            cfg["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            cfg["top_k"] = self.config.top_k
        if json_mode or response_schema is not None:
            cfg["response_mime_type"] = "application/json"
            if response_schema is not None:
                if not isinstance(response_schema, dict):
                    raise TypeError(
                        "response_schema must be a dict (JSON Schema).",
                    )
                cfg["response_schema"] = response_schema
        return cfg

    def _generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        json_mode: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call Gemini generate_content and return text."""
        contents = self._to_contents(messages)
        cfg = self._config(
            max_tokens,
            temperature,
            json_mode,
            response_schema,
        )
        resp = self._client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=cfg,
        )
        return _extract_text(resp)

    @retry(
        retry=retry_if_exception_type(GeminiRateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *args,
        **kwargs
    ) -> str:
        """Chat with Gemini. kwargs: json_mode, response_schema."""
        json_mode = kwargs.pop("json_mode", False)
        response_schema = kwargs.pop("response_schema", None)
        try:
            return self._generate(
                messages,
                max_tokens,
                temperature,
                json_mode=json_mode,
                response_schema=response_schema,
            )
        except TypeError:
            raise
        except Exception as e:
            s = str(e).lower()
            if "rate_limit" in s or "quota" in s:
                raise GeminiRateLimitError("Rate limit exceeded")
            raise GeminiError(f"Gemini API error: {e}")

    def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *args,
        **kwargs
    ) -> str:
        """Complete via chat with single user message."""
        logger.debug(f"Complete: {len(prompt)} chars")
        out = self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            *args,
            **kwargs,
        )
        logger.debug(f"Response: {len(out)} chars")
        return out

    def embed(self, text: str) -> List[float]:
        """Not supported; use Vertex AI Embeddings."""
        raise NotImplementedError(
            "Gemini has no embedding endpoint. "
            "Use Vertex AI Embeddings API.",
        )

    def list_models(self) -> List[str]:
        """Return current recommended Gemini model names."""
        return [
            # Gemini 3 series (latest, preview)
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            # Gemini 2.5 series (stable, recommended)
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            # Gemini 2.0 series (previous generation)
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]
