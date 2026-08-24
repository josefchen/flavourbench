# Contributing

FlavourBench treats benchmark changes as scientific changes. A pull request that changes tasks,
model routes, score semantics, or reported results should include:

1. the precise research question;
2. the changed artifact and schema versions;
3. deterministic reproduction instructions;
4. tests for scoring, missingness, and provenance behavior;
5. an explanation of whether old and new scores remain comparable; and
6. confirmation that no credential, private database, participant data, or unrestricted Epicure
   payload is included.

Before pushing any change, run the same provider-free gate used by GitHub Actions:

```bash
make format
make ci
```

`make ci` verifies the statistical release, runs the public test suite, reconstructs the lab
dataset, checks the PDF and arXiv bundle, enforces Ruff formatting, and scans the public tree for
private paths and credential patterns. Generated release artifacts must be content-addressed and
reproducible from checked-in public inputs.

Please keep the public automated track separate from any future human-judgment track. Do not
reinterpret unavailable endpoints as low model capability, or silently replace a model route.
