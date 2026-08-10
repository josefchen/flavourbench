# Epicure MCP 1790 R1 — release-candidate model card

`Epicure MCP 1790 R1` is the proposed immutable identity for the currently unmatched 1,790 × 300
Epicure representation used by the MCP runtime. It is a recovered artifact, not an established
copy of the published Cooc, Core, Chem, or Paper II model. The exact bundle SHA-256 is
`98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1`.

## Intended benchmark role

FlavourBench may treat R1 as an opaque, content-addressed evidence intervention. The benchmark can
estimate the effect of giving an LLM access to this exact frozen system. It cannot infer how the
embedding was trained, attribute it to a public sibling, or use Epicure similarity as its own
independent ground truth.

## Known limitations

- Exact training run, seed, training source revision, and build environment are not recovered.
- The current application identity is a development-source manifest, not a clean release or OCI
  digest.
- The current deployed service has not yet attested the required provenance endpoint.
- At least one response still reports `model: cooc`; official R1 must report the distinct R1 ID.
- Code licensing does not establish payload rights; a payload license and signed artifact-rights
  attestation remain required.
- Algorithm-version maps, parity fixtures, independent reproduction, and data-steward
  certifications remain missing. A platform-scoped dependency lock, CycloneDX SBOM, and verified
  private offline rebuild now exist, but they do not recover training lineage or payload rights.

Accordingly, the adjacent release-candidate manifest is `rank_eligible: false`. A later official
runtime contract must supersede it after every release gate is resolved; this file must never be
edited to pretend the prerequisites already passed.

## Recovered inventory update — 3 August 2026

The append-only recovery inventory
`epicure-recovered-runtime-inventory-70d00d933aa1340841a82a9637de8b75de380f8aeba2179beab419fb6542ab5f.json`
now enumerates every available bundled-data and Python source file and records the observed local
environment. An authenticated local runtime attestation matched bundle
`98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1` and application
`be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313` exactly. This improves exact
retrospective attribution but does not change the blocked release status: the source checkout is
dirty, the public provenance route is not deployed, training lineage is unrecovered, and payload
rights remain unattested. See `EPICURE-MCP-1790-R1-REPRODUCIBILITY.md`.

The private reconstruction supplement binds the exact source and data manifests to 41 hash-locked
runtime wheels and a CycloneDX 1.5 SBOM. A clean no-index reinstall reproduced the source, payload,
and package runtime identities. Authority record `83b5f310...` preserves the boundary: this was a
same-operator private rebuild, no public payload is provided, no immutable OCI digest was produced,
and the runtime remains `rank_eligible: false`.

This digest replaces the recovery inventory cited by current prose and release packaging. The
earlier `ef865046...` file remains intact because it is referenced by frozen v1 records, but its
first Git status path was parsed incorrectly. The correction changes no Epicure runtime identity.
