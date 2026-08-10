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

Code-only improvements should run the relevant test files and Ruff checks. Paper changes should
also run `make -C paper verify`. Generated release artifacts must be content-addressed and must be
reproducible from checked-in public inputs.

Please keep the public automated track separate from any future human-judgment track. Do not
reinterpret unavailable endpoints as low model capability, or silently replace a model route.
