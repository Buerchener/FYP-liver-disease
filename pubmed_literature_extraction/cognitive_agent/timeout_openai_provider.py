"""Bounded LangExtract OpenAI provider for unreliable compatible proxies."""

from __future__ import annotations

import httpx
from langextract.providers import router
from langextract.providers.openai import OpenAILanguageModel
from openai import OpenAI


@router.register(r"^liverkg_timeout_openai$", priority=100)
class LiverKGTimeoutOpenAIModel(OpenAILanguageModel):
    """Use a real HTTP deadline instead of the SDK's unbounded default."""

    def __init__(
        self,
        *args,
        connect_timeout_s: float = 20.0,
        request_timeout_s: float = 180.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._client.close()
        timeout = httpx.Timeout(
            connect=max(1.0, float(connect_timeout_s)),
            read=max(1.0, float(request_timeout_s)),
            write=max(1.0, float(request_timeout_s)),
            pool=max(1.0, float(connect_timeout_s)),
        )
        self._http_client = httpx.Client(
            timeout=timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
        )
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            max_retries=0,
            timeout=timeout,
            http_client=self._http_client,
        )

    def close(self) -> None:
        self._client.close()
        self._http_client.close()
