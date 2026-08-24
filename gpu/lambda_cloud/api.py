from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

BASE_URL = "https://cloud.lambda.ai/api/v1"


class LambdaCloudError(RuntimeError):
    """Raised when Lambda Cloud rejects or cannot complete an API request."""


class LambdaCloudClient:
    def __init__(self, api_key: str, *, timeout_seconds: int = 30) -> None:
        if not api_key:
            raise LambdaCloudError("LAMBDA_API_KEY is required for provider queries")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        if not path.startswith("/") or ".." in path:
            raise LambdaCloudError("invalid Lambda Cloud API path")
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        request = urllib.request.Request(  # noqa: S310 - BASE_URL is a fixed HTTPS origin.
            f"{BASE_URL}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Epicure-FlavourBench-GPU-Controller/0.1",
            },
        )
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=self.timeout_seconds
            ) as response:
                document = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:1000]
            raise LambdaCloudError(f"Lambda Cloud HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LambdaCloudError(f"Lambda Cloud request failed: {type(exc).__name__}") from exc
        if not isinstance(document, dict) or "data" not in document:
            raise LambdaCloudError("Lambda Cloud returned an unexpected response")
        return document["data"]

    def instance_types(self) -> dict[str, Any]:
        data = self.request("GET", "/instance-types")
        if not isinstance(data, dict):
            raise LambdaCloudError("instance-types data must be an object")
        return data

    def launch(self, payload: dict[str, Any]) -> list[str]:
        data = self.request("POST", "/instance-operations/launch", payload)
        instance_ids = data.get("instance_ids", []) if isinstance(data, dict) else []
        if not isinstance(instance_ids, list) or not all(
            isinstance(value, str) for value in instance_ids
        ):
            raise LambdaCloudError("launch did not return instance IDs")
        return instance_ids

    def instance(self, instance_id: str) -> dict[str, Any]:
        data = self.request("GET", f"/instances/{instance_id}")
        if not isinstance(data, dict):
            raise LambdaCloudError("instance data must be an object")
        return data

    def instances(self) -> list[dict[str, Any]]:
        data = self.request("GET", "/instances")
        if not isinstance(data, list):
            raise LambdaCloudError("instances data must be an array")
        return [value for value in data if isinstance(value, dict)]

    def terminate(self, instance_ids: list[str]) -> list[dict[str, Any]]:
        data = self.request(
            "POST", "/instance-operations/terminate", {"instance_ids": instance_ids}
        )
        terminated = data.get("terminated_instances", []) if isinstance(data, dict) else []
        if not isinstance(terminated, list):
            raise LambdaCloudError("termination returned an unexpected response")
        return [value for value in terminated if isinstance(value, dict)]
