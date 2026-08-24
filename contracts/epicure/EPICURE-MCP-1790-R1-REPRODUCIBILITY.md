# Epicure MCP 1790 R1 reproducibility boundary

This document records what can and cannot be reproduced for the Epicure intervention used by the
current FlavourBench development study. It does not promote the intervention to an official model
release.

## Recovered runtime identity

The append-only inventory
`epicure-recovered-runtime-inventory-70d00d933aa1340841a82a9637de8b75de380f8aeba2179beab419fb6542ab5f.json`
binds the following runtime:

- runtime ID: `epicure-mcp-1790-r1+bundle.98d0403115bf.app.be4216ae799f`;
- data bundle SHA-256: `98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1`;
- application source-manifest SHA-256:
  `be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313`;
- semantic MCP tool-catalog SHA-256:
  `666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd`;
- shape: 1,790 ingredients by 300 embedding dimensions.

The inventory contains all eleven bundled-data file names, byte counts, and SHA-256 values; all 28
Python source-file names, byte counts, and SHA-256 values; the relevant build and policy file
hashes; the Git HEAD and dirty-file manifest; and a package/version snapshot of the observed local
environment. The recovery inventory itself remains append-only. A later reconstruction supplement
now binds that observed environment to a hash-locked dependency set and SBOM.

All eleven exact data files are also publicly readable under `data/` at immutable Git commit
`14ddf04aba81a76b75efa6554041f6bff48992c6` in
[`KAIKAKU-AI/epicure-mcp`](https://github.com/KAIKAKU-AI/epicure-mcp/tree/14ddf04aba81a76b75efa6554041f6bff48992c6/data).
A byte-for-byte audit on 3 August 2026 matched every file's recorded size and SHA-256 to the runtime
manifest. This closes public technical availability of the data bytes. It does not establish a
payload licence, redistribution authority, or the rights of any upstream corpus source.

Authenticated read-only requests to loopback runtimes at `127.0.0.1:8081/provenance` and
`127.0.0.1:18082/provenance` matched the recovered bundle, application, shape, and exploratory
runtime label exactly. The paid reasoning-effort smokes were bound to the latter runtime. The public
`https://epicure-mcp.kaikaku.ai/provenance` route returned HTTP 404 on 3 August 2026. Therefore the
current public service does not yet provide independent provenance verification.

## Verification

From the evaluation repository root:

```bash
set -a
. "${EPICURE_ROOT:?set EPICURE_ROOT}/.env"
set +a
cd "${EPICURE_MCP_ROOT:?set EPICURE_MCP_ROOT}"
PORT=18082 .venv/bin/python -m epicure_mcp.server
```

In a second terminal, capture the authenticated response and inventory it without writing the
bearer token into either artifact:

```bash
set -a
. "${EPICURE_ROOT:?set EPICURE_ROOT}/.env"
set +a
cd "${FLAVOURBENCH_RESEARCH_ROOT:?set FLAVOURBENCH_RESEARCH_ROOT}"
PYTHONPATH=flavourbench/src flavourbench/.venv/bin/python \
  -m flavourbench.epicure_lineage_inventory \
  --mcp-root "$EPICURE_MCP_ROOT" \
  --tool-contract \
  flavourbench/contracts/epicure/tool-catalog-666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd.json \
  --runtime-provenance-url http://127.0.0.1:18082/provenance \
  --runtime-token-env MCP_API_TOKEN \
  --output-dir \
  flavourbench/artifacts/season1/epicure-lineage/local-attested-20260803-18082-corrected
```

The generated artifact digest must be
`70d00d933aa1340841a82a9637de8b75de380f8aeba2179beab419fb6542ab5f` when run against the exact
observed checkout, environment, and attestation. A later clean release should produce a different
artifact and must supersede this record rather than editing it.

The earlier inventory with digest
`ef865046abe13fc4260d9c2a1981e4ddf6ec44195e49ddd0ff549cf56a5b75e2` remains as an immutable
historical record. Its Git porcelain parser stripped the leading status-space from the first entry,
recording `Dockerfile` as `ockerfile` with no byte count or hash. The corrected inventory parses the
NUL-delimited status fields and records `Dockerfile` as 1,376 bytes with SHA-256
`af17c9d344147c82cb7e00e426ebfd1d028ba60abaeb21a1cac0e955708e1461`. No runtime, bundle,
application, tool, environment, or attestation field changed. New plans and paper archives must cite
the corrected digest; old append-only plans retain the historical digest.

Correction record
`epicure-lineage-inventory-correction-d739a1b08be79c8a116ea86687ef8f4c983fe8cc9c312257f85ef107e32e90e7.json`
marks `70d00d...` as the authoritative recovery inventory and `ef865046...` as a retained,
non-authoritative historical record. New execution plans must bind both the authoritative inventory
and this correction record.

The captured response is itself stored as
`runtime-provenance-attestation-825d087713ef525a1643a81bb9b94c26f3be64794ac84b4699d1cc8380922220.json`.

## Verified private runtime reconstruction

The authority record
`epicure-runtime-reconstruction-authority-83b5f3109242478f6def0acbc434112900760ae92890626da23838bac0ea5a6a.json`
points to the following authoritative supplement:

- exact source/data/runtime manifest:
  `a37e5d25f9c5f7a1ec32708b17e0301bbd88248b4c0aeacecf89579106d8edf5`;
- observed-environment dependency lock SHA-256:
  `86fce704f665270d18a48812b489c651efc8a5688637fa7e89fcd641b9b8d5f1`;
- CycloneDX 1.5 SBOM SHA-256:
  `ec689f51124f307dcaeb1de33007dd6065b97275a9f561a1a84b1bcad97fc25b`;
- private no-index rebuild receipt:
  `35854e9f50f8f3756ab480a6ded012e15bb2e2cc4673948923068cc9deb88255`.

The lock pins 41 CPython 3.12, Linux x86_64 runtime wheels to exact versions and SHA-256 values.
The verifier checked every installed distribution version and every physical file carrying a
SHA-256 entry in its package RECORD. It then installed the sealed wheelhouse into a clean temporary
environment with `pip --no-index --require-hashes`, ran the data-shape verifier, and reproduced the
same bundle and application identities. The observed and rebuilt importable runtime payloads have
the same aggregate manifest SHA-256,
`f715e57d4c8879916d56aeb6c0983c75ed9f6ad7a96187906cfbb375e8e49fd0`.

This is a verified private reconstruction by the same operator from the same checkout. It is not an
independent reproduction. The wheelhouse and payload are not included in the research archive.
The earlier `0735af49...` manifest and `602f9d31...` receipt remain historical, non-authoritative
records because their resolver selected dependency versions newer than those observed in the
loopback runtime.

## Public verification packet

The content-addressed public packet in
`artifacts/season1/epicure-lineage/public-reconstruction/` turns the archive boundary into a
machine-checkable record. It lists each published inventory, attestation, manifest, lock, SBOM,
recipe, verifier, and lineage-falsification record with both its repository path and portable arXiv
archive path, byte count, and SHA-256. Packet v3 binds the immutable public Git commit, raw URL,
byte count, and SHA-256 for each of the eleven exact data files. The data are not mirrored inside
the research archive. It also includes a deterministic 35,176-byte source-only archive with
SHA-256 `d08fb475e9c325a8c41daf5b789e6b4bca547228139eece4578f9b06c324703c`.
That archive has exactly 30 regular members: the 28 studied Python files, a machine-readable
manifest, and the existing 1,067-byte KAIKAKU-AI MIT `LICENSE`. Member metadata is normalized to
mode 0644, zero timestamps, numeric owner/group zero, and empty owner/group names. It contains no
runtime data, model payload, dependency wheels, credentials, or training corpus and introduces no
new licence.

The remaining omission register covers the private wheelhouse, original training input, corpus
payloads, payload-rights evidence, signed release, immutable image, and independent reproduction.

The source comparison remains explicit. Twenty-one of the 28 exact study application files match
the public Git commit. Six public paths contain different bytes and one path is absent there:

- `src/epicure_mcp/config.py` — `6221163a4d0afe793b62b876e34e8a6f6af056a4194527a826ed9ecf9bccfbfe`;
- `src/epicure_mcp/geometry.py` — `152d0b6f5da863b5f2ddbcefa7176a87eb9c90c58dbaea83a0ace4d024b6846f`;
- `src/epicure_mcp/security.py` — `437205948f4a3610353cc500a52fbe96ddea00c52a2da5119bad19774ab44fed`;
- `src/epicure_mcp/server.py` — `83ff0f10bb39c3a08d6fc96f944f010df21e0baf25020928ff58147598d4b693`;
- `src/epicure_mcp/tools/find_pairings.py` — `11e634f793f98fab9ebabf549d10899b8c9b10881c84889fac7e8a33f84854b6`;
- `src/epicure_mcp/tools/morph.py` — `544d64f4325443637734b5f56d9a2536b5698f45a10b140e0aaad958e64c1e97`;
- `src/epicure_mcp/provenance.py` — `822b4e2a3af78b9cca76548d11654998e90e438f195cdf62c0ed363aa5d44223`
  (absent from the reference commit).

Those seven exact study blobs are now present in the source-only research archive. This closes
application-byte availability, not the clean signed-release or exact-OCI gates.

Rebuild the source archive from the preserved exact checkout:

```bash
cd "${FLAVOURBENCH_RESEARCH_ROOT:?set FLAVOURBENCH_RESEARCH_ROOT}"
PYTHONPATH=flavourbench/src flavourbench/.venv/bin/python \
  -m flavourbench.epicure_public_reconstruction build-source-archive \
  --root flavourbench \
  --mcp-root "$EPICURE_MCP_ROOT" \
  --output-dir flavourbench/artifacts/season1/epicure-lineage/public-reconstruction
```

The resulting filename must end in
`d08fb475e9c325a8c41daf5b789e6b4bca547228139eece4578f9b06c324703c.tar.gz`.

Verify the repository layout without a network call:

```bash
cd "${FLAVOURBENCH_RESEARCH_ROOT:?set FLAVOURBENCH_RESEARCH_ROOT}"
PYTHONPATH=flavourbench/src flavourbench/.venv/bin/python \
  -m flavourbench.epicure_public_reconstruction verify \
  --packet flavourbench/artifacts/season1/epicure-lineage/public-reconstruction/\
epicure-public-reconstruction-packet-*.json \
  --root flavourbench \
  --layout repository
```

After extracting the arXiv source archive, the same verifier is standalone:

```bash
python provenance/epicure-public-reconstruction-verifier.py verify \
  --packet provenance/epicure-public-reconstruction-packet.json \
  --root . \
  --layout archive
```

If the eleven files have already been downloaded or checked out, verify them without a network
call by pointing the verifier at the directory containing them:

```bash
python provenance/epicure-public-reconstruction-verifier.py verify \
  --packet provenance/epicure-public-reconstruction-packet.json \
  --root . --layout archive \
  --runtime-data-dir /path/to/epicure-mcp/data
```

Network access is opt-in. This command streams the immutable raw GitHub URLs and checks all eleven
sizes and hashes without saving a second copy:

```bash
python provenance/epicure-public-reconstruction-verifier.py verify \
  --packet provenance/epicure-public-reconstruction-packet.json \
  --root . --layout archive \
  --verify-runtime-data-online
```

The check fails on a changed byte count or hash, a malformed content address, a path traversal,
credential-like material, an inconsistent runtime identity, an accepted candidate training
lineage, or any attempt to mark a closed release gate as passed. A successful check means the
published metadata and every exact application-source archive member are intact. With either
optional data mode, it also means the observed data bytes match the frozen runtime manifest. The
research archive alone still lacks the runtime data and private wheelhouse. The packet does not
establish payload rights, create a clean signed release, identify the studied OCI image, or
constitute an independent rebuild.

## Facts that remain unrecovered

The following facts are absent and must not be inferred from vocabulary overlap, geometric
similarity, file names, or the `model: cooc` response field:

- embedding-training run ID, seed, code revision, and complete environment;
- a source-to-artifact rights matrix for the training corpus;
- an exact match to the public Cooc, Core, Chem, or Paper II artifacts;
- a signed release tag and immutable OCI image digest;
- an independent rebuild demonstrating parity from licensed sources.

The local Git checkout is dirty. Its exact working-tree source identity is captured, but Git commit
`5feb9fb80a26d59614e1fb80fcdac76e8fa58eb9` alone does not identify the application used by the
study.

## Redistribution boundary

The repository records an MIT license for the application code, and the exact data bytes are
publicly readable at the immutable commit above. Neither fact, by itself, establishes that the
model payload or every upstream corpus component may be redistributed by the study. Until the data
steward supplies a signed payload-rights attestation and source-rights matrix, the payload remains
`redistributable: false` in this release record. Public releases may distribute this inventory, the
hashes, aggregate benchmark statistics, and code whose license is established; they must not infer
or attach a payload licence from public availability alone.

## Gates for a superseding release

A superseding official runtime contract requires all of the following:

1. a clean, signed application release and immutable OCI digest;
2. the public `/provenance` endpoint returning the exact release, bundle, application, and tool
   identities used by the benchmark;
3. correction of misleading public-sibling identity fields;
4. a signed payload-rights attestation and source-rights matrix;
5. golden tool fixtures and an independent parity reproduction;
6. either recovered training lineage or an explicit, approved opaque-artifact release boundary.

The hash-locked dependency environment, CycloneDX SBOM, exact source/data manifest, and private
offline rebuild are now evidenced. They do not close any of the remaining gates above.

Until then, the recovered runtime remains usable only as a precisely attributed, opaque development
intervention. It is not rank eligible.
