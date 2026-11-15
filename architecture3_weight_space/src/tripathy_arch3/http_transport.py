"""Small injectable HTTP boundary used by provider adapters and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> object:
        return httpx.Response(self.status_code, content=self.body).json()


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: object | None = None,
        timeout_seconds: float = 20.0,
    ) -> HttpResponse: ...


class HttpxTransport:
    """Production transport. Provider credentials are never returned or logged."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: object | None = None,
        timeout_seconds: float = 20.0,
    ) -> HttpResponse:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=False) as client:
            response = client.request(method, url, headers=headers, json=json_body)
        return HttpResponse(
            status_code=response.status_code,
            headers={key.lower(): value for key, value in response.headers.items()},
            body=response.content,
        )
