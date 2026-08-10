from types import SimpleNamespace

import pytest

from flavourbench import service_qwencloud
from flavourbench.provider import ProviderError
from flavourbench.qwencloud_catalog import QwenCloudCatalogError
from flavourbench.service_qwencloud import QwenCloudDirectProvider


def test_direct_provider_rejects_token_plan_key_before_client_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        qwencloud_api_key="sk-sp-test-credential-not-real",
        qwencloud_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setattr(service_qwencloud, "get_settings", lambda: settings)

    with pytest.raises(ProviderError, match="Token Plan and Coding Plan credentials"):
        QwenCloudDirectProvider()


def test_direct_provider_rejects_token_plan_host_before_client_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        qwencloud_api_key="sk-ws-test-credential-not-real",
        qwencloud_base_url=(
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        ),
    )
    monkeypatch.setattr(service_qwencloud, "get_settings", lambda: settings)

    with pytest.raises(QwenCloudCatalogError, match="pay-as-you-go"):
        QwenCloudDirectProvider()


def test_direct_provider_accepts_legacy_pay_as_you_go_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        qwencloud_api_key="sk-test-legacy-payg-not-real",
        qwencloud_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    base_init_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(service_qwencloud, "get_settings", lambda: settings)
    monkeypatch.setattr(
        service_qwencloud.KimiDirectProvider,
        "__init__",
        lambda _self, *args, **kwargs: base_init_calls.append((args, kwargs)),
    )

    QwenCloudDirectProvider()

    assert base_init_calls == [((), {})]
