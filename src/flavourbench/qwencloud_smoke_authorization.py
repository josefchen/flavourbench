"""Record one exact Human-PI GO after the QwenCloud no-provider preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .qwencloud_smoke_admission import (
    QwenCloudSmokeAdmissionError,
    _regular_json,
    _verified_artifact,
    _write_content_addressed,
    build_human_pi_authorization,
    verify_go_template,
    verify_preflight_artifact,
)


def authorize(args: argparse.Namespace) -> Path:
    template = verify_go_template(
        args.go_template,
        expected_sha256=args.expected_go_template_sha256,
    )
    preflight = verify_preflight_artifact(
        args.preflight,
        expected_sha256=args.expected_preflight_sha256,
        template=template,
    )
    standing = _verified_artifact(
        args.human_pi_identity_record,
        expected_sha256=args.expected_human_pi_identity_record_sha256,
        label="standing Human-PI identity record",
    )
    authorization = build_human_pi_authorization(
        template=template,
        preflight=preflight,
        standing_human_pi_record=standing,
        confirmation=args.confirm,
        recorded_at=args.recorded_at,
    )
    return _write_content_addressed(
        args.output_dir,
        "qwencloud-one-pair-human-pi-go",
        authorization,
    )


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go-template", type=Path, required=True)
    parser.add_argument("--expected-go-template-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--human-pi-identity-record", type=Path, required=True)
    parser.add_argument("--expected-human-pi-identity-record-sha256", required=True)
    parser.add_argument("--recorded-at", required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        path = authorize(args)
        artifact = _regular_json(path, label="QwenCloud Human-PI GO")
    except (OSError, QwenCloudSmokeAdmissionError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "status": "exact_one_pair_human_pi_go_recorded_no_external_calls",
                "provider_calls_made": False,
                "epicure_calls_made": False,
                "artifact": str(path.resolve()),
                "artifact_sha256": artifact["artifact_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
