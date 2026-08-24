"""Bedrock Mantle-primary routing with explicit OpenRouter substitution.

Fallback is allowed only when Mantle proves that no inference began.  The
OpenRouter backend must bind the exact same FlavourBench canonical model.  A
fallback result is always provider-substituted, unranked, and unpooled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .bedrock_mantle import (
    MantleGenerationResult,
    MantleGenerationSpec,
    MantleRouteUnavailable,
)


class MantlePrimaryBackend(Protocol):
    async def generate(self, spec: MantleGenerationSpec) -> MantleGenerationResult: ...


@dataclass(frozen=True)
class MantleOpenRouterFallbackResult:
    canonical_model_id: str
    actual_model_id: str
    provider_slug: str
    payload: Any


class MantleOpenRouterFallbackBackend(Protocol):
    canonical_model_id: str

    async def generate(self, spec: MantleGenerationSpec) -> MantleOpenRouterFallbackResult: ...


@dataclass(frozen=True)
class MantleRoutedGenerationResult:
    canonical_model_id: str
    route: str
    provider_substitution: bool
    rank_eligible: bool
    unpooled: bool
    pooling_group: str
    mantle_result: MantleGenerationResult | None
    openrouter_result: MantleOpenRouterFallbackResult | None
    fallback_reason: str | None = None


class MantlePrimaryRouter:
    def __init__(
        self,
        primary: MantlePrimaryBackend,
        *,
        canonical_model_id: str,
        fallback: MantleOpenRouterFallbackBackend | None = None,
        allow_openrouter_fallback: bool = False,
    ) -> None:
        if not canonical_model_id:
            raise ValueError("Mantle routing requires a canonical model ID")
        if fallback is not None and fallback.canonical_model_id != canonical_model_id:
            raise ValueError("OpenRouter fallback must bind to exactly the same canonical model")
        if allow_openrouter_fallback and fallback is None:
            raise ValueError("enabled OpenRouter fallback requires a backend")
        self.primary = primary
        self.canonical_model_id = canonical_model_id
        self.fallback = fallback
        self.allow_openrouter_fallback = allow_openrouter_fallback

    async def generate(self, spec: MantleGenerationSpec) -> MantleRoutedGenerationResult:
        if spec.canonical_model_id != self.canonical_model_id:
            raise ValueError("request and Mantle route canonical model IDs differ")
        try:
            result = await self.primary.generate(spec)
        except MantleRouteUnavailable as error:
            if not self.allow_openrouter_fallback or self.fallback is None:
                raise
            fallback_result = await self.fallback.generate(spec)
            if fallback_result.canonical_model_id != self.canonical_model_id:
                raise ValueError("fallback returned a different canonical model") from error
            if fallback_result.actual_model_id != self.canonical_model_id:
                raise ValueError("fallback returned a different actual model") from error
            return MantleRoutedGenerationResult(
                canonical_model_id=self.canonical_model_id,
                route="openrouter_fallback",
                provider_substitution=True,
                rank_eligible=False,
                unpooled=True,
                pooling_group="unranked_provider_substitution",
                mantle_result=None,
                openrouter_result=fallback_result,
                fallback_reason=str(error),
            )

        if result.identity.canonical_model_id != self.canonical_model_id:
            raise ValueError("Mantle primary returned a different canonical model")
        return MantleRoutedGenerationResult(
            canonical_model_id=self.canonical_model_id,
            route="bedrock_mantle_primary",
            provider_substitution=False,
            rank_eligible=result.rank_eligible,
            unpooled=False,
            pooling_group="bedrock_mantle_primary",
            mantle_result=result,
            openrouter_result=None,
        )
