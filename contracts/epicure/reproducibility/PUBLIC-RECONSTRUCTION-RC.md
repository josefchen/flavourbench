# Epicure public-input reconstruction candidate

This kit reconstructs the exact Epicure application and runtime-data bytes used by the
FlavourBench development study from content-addressed source material. It is a technical
reconstruction aid, not an Epicure model release or a licence grant.

## What the kit contains

- the 28-file exact study application archive and its existing MIT licence;
- the exact source/data/runtime manifest;
- a 41-wheel, SHA-256-locked CPython 3.12 Linux x86-64 dependency set;
- a CycloneDX 1.5 dependency SBOM;
- the frozen MCP tool catalog;
- immutable GitHub URLs, byte counts, and SHA-256 values for all eleven runtime-data files;
- same-operator functional fixtures for provenance, `pairing_score`, and `neighbors`;
- a fail-closed, standard-library reconstruction and verification program; and
- an explicit rights and lineage boundary.

The kit deliberately contains no model/data payload bytes, wheel files, credentials, training
corpus, training input embedding, or OCI image. Runtime data are fetched only after an explicit
technical-verification acknowledgement. Dependencies are fetched from the public PyPI index and
must match the frozen hashes before installation. After installation, every declared wheel RECORD
hash is checked against the physical file and the importable runtime-payload aggregate must match
the previously sealed environment.

## Target

The executable path is intentionally narrow: CPython 3.12 on Linux x86-64 with a
manylinux-compatible glibc. A different operating system, architecture, or Python minor version
is outside this reconstruction claim.

## Verify the kit

After extracting it into an empty directory:

```bash
python3.12 reconstruct.py verify-kit --kit-root .
```

This command is offline. It verifies the kit manifest, every registered member, the runtime
manifest, the dependency lock, the CycloneDX SBOM, the source archive, the data-source map, the
functional fixture, and the fail-closed rights record.

## Reconstruct from public inputs

Network access is opt-in. The following command downloads exact data bytes from the frozen Git
commit and exact wheels from `https://pypi.org/simple`, installs the wheels into a clean virtual
environment, checks all source/data/dependency hashes, executes the frozen functional fixtures,
and writes a content-addressed receipt:

```bash
python3.12 reconstruct.py materialize \
  --kit-root . \
  --workspace /path/to/empty/epicure-reconstruction \
  --online \
  --with-dependencies \
  --run-probe \
  --acknowledge-unattested-payload-rights
```

The acknowledgement means only that the operator understands that public availability is not a
payload licence. It does not waive or establish any right.

An operator with previously acquired inputs may instead use `--data-dir` and `--wheelhouse`; the
program hashes them before use and makes no network request unless `--online` is supplied.

## Claims this can support

A successful receipt supports these narrow technical claims:

1. the reconstructed source and data bytes have the exact identities recorded for the study;
2. the installed dependency wheels match the frozen 41-wheel hash set;
3. the reconstructed runtime reports the expected 1,790 by 300 shape and application/data hashes;
4. two small deterministic tool fixtures reproduce the same outputs; and
5. the reconstruction used only the kit plus explicitly fetched public inputs.

It does **not** make the operator independent of the study authors, recover how the embedding was
trained, establish corpus or payload rights, identify an immutable OCI image, convert the dirty
study checkout into a signed release, or make the runtime eligible for an official ranking.

## Release-candidate checklist

The following gates remain prerequisites for a superseding official Epicure release:

- signed payload-rights attestation and source-by-source rights matrix;
- recovered training run, seed, code, environment, and input lineage, or a formally approved
  opaque-artifact release boundary;
- clean signed source release and immutable OCI digest;
- public `/provenance` serving the exact studied identities;
- execution and fixture parity receipt from an operator independent of the study authors; and
- governance approval of the resulting release for official benchmark use.

Do not reinterpret the same-operator fixture or reconstruction receipt as satisfying the
independent-reproduction gate.
