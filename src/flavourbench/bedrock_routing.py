"""Bedrock-primary routing with explicit, unpooled OpenRouter substitution.

Only an explicit pre-generation ``BedrockRouteUnavailable`` may trigger the
fallback. Invalid output, tool errors, uncertain network failures, and other
post-acceptance failures remain Bedrock failures and are never cherry-picked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .bedrock_provider import (
    BedrockGenerationResult,
    BedrockGenerationSpec,
    BedrockRouteUnavailable,
)


class BedrockPrimaryBackend(Protocol):
    async def generate(self, spec: BedrockGenerationSpec) -> BedrockGenerationResult: ...


@dataclass(frozen=True)
class OpenRouterFallbackResult:
    canonical_model_id: str
    actual_model_id: str
    provider_slug: str
    payload: Any


class OpenRouterFallbackBackend(Protocol):
    canonical_model_id: str

    async def generate(self, spec: BedrockGenerationSpec) -> OpenRouterFallbackResult: ...


@dataclass(frozen=True)
class RoutedGenerationResult:
    canonical_model_id: str
    route: str
    provider_substitution: bool
    rank_eligible: bool
    unpooled: bool
    pooling_group: str
    bedrock_result: BedrockGenerationResult | None
    openrouter_result: OpenRouterFallbackResult | None
    fallback_reason: str | None = None


class BedrockPrimaryRouter:
    def __init__(
        self,
        primary: BedrockPrimaryBackend,
        *,
        canonical_model_id: str,
        fallback: OpenRouterFallbackBackend | None = None,
        allow_openrouter_fallback: bool = False,
    ) -> None:
        if not canonical_model_id:
            raise ValueError("routing requires a canonical model ID")
        if fallback is not None and fallback.canonical_model_id != canonical_model_id:
            raise ValueError("OpenRouter fallback must bind to exactly the same canonical model")
        if allow_openrouter_fallback and fallback is None:
            raise ValueError("enabled OpenRouter fallback requires a backend")
        self.primary = primary
        self.canonical_model_id = canonical_model_id
        self.fallback = fallback
        self.allow_openrouter_fallback = allow_openrouter_fallback

    async def generate(self, spec: BedrockGenerationSpec) -> RoutedGenerationResult:
        if spec.canonical_model_id != self.canonical_model_id:
            raise ValueError("request and routing canonical model IDs differ")
        try:
            result = await self.primary.generate(spec)
        except BedrockRouteUnavailable as error:
            if not self.allow_openrouter_fallback or self.fallback is None:
                raise
            fallback_result = await self.fallback.generate(spec)
            if fallback_result.canonical_model_id != self.canonical_model_id:
                raise ValueError("fallback returned a different canonical model") from error
            return RoutedGenerationResult(
                canonical_model_id=self.canonical_model_id,
                route="openrouter_fallback",
                provider_substitution=True,
                rank_eligible=False,
                unpooled=True,
                pooling_group="unranked_provider_substitution",
                bedrock_result=None,
                openrouter_result=fallback_result,
                fallback_reason=str(error),
            )

        return RoutedGenerationResult(
            canonical_model_id=self.canonical_model_id,
            route="bedrock_primary",
            provider_substitution=False,
            rank_eligible=result.rank_eligible,
            unpooled=False,
            pooling_group="bedrock_primary",
            bedrock_result=result,
            openrouter_result=None,
        )
