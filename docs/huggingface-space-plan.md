# Hugging Face publication specification

## Product position

FlavourBench is the executable culinary benchmark for frontier language models. The public
product has three jobs:

1. rank models on one complete, shared task matrix;
2. expose enough evidence to audit any score; and
3. let a lab evaluate or train its own model without sending credentials to FlavourBench.

The Space is a scientific instrument, not a promotional landing page. It opens with the result,
then provides discovery, uncertainty, evidence, execution, comparison, and method in that order.

## Reference products

The publication design was checked against current benchmark products on 23 August 2026.

| Product | Pattern retained in FlavourBench |
|---|---|
| [LiveBench](https://livebench.ai) | Dedicated insights view, sortable evidence, release context |
| [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) | Search, fast scope filters, downloadable results |
| [SWE-bench](https://www.swebench.com) | Explicit submission contract, setup disclosure, versioned result history |
| [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | Clear navigation across results, models, tasks, and comparison |
| [LMArena](https://huggingface.co/spaces/lmarena-ai/arena-leaderboard) | Confidence-aware ranking and domain views |
| [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) | Dense research table, update context, capability breakdown |

The visual treatment does not copy any of them. It uses the same identity as the launch figures.

## Information architecture

| Surface | Question answered |
|---|---|
| Leaders | Which models score highest, and what is the complete point order? |
| Insights | Which score gaps are resolved, and where do leading labs differ? |
| Profiles | What is one model's substitution, pairing, and constraint profile? |
| Inspect | Which exact prompt, response, reward map, route, and hashes produced a cell? |
| Run your model | How can a lab run, score, train, and submit a result? |
| Compare | Is one selected pair distinguishable on the shared tasks? |
| Method | What is the scoring contract, inference procedure, and training boundary? |

## Visual contract

The interface matches the launch assets:

| Token | Value |
|---|---|
| Paper | `#F6F7F5` |
| Ink | `#161817` |
| Muted text | `#68706C` |
| Structural line | `#56605B` |
| Rule | `#DDE1DE` |
| Launch red | `#A83D34` |
| Display and body | Lato |
| Measurements and hashes | DejaVu Sans Mono |

Rules are sharp. There are no gradients, glows, floating cards, decorative borders, or ornamental
motion. Red is reserved for the leader, resolved emphasis, and interaction focus. Dark mode keeps
the same hierarchy. Mobile layouts show the primary score columns directly and move detailed
intervals into the Insights view.

Design controls: variance 5, motion 2, information density 8.

## Publication architecture

Hugging Face recommends separating the leaderboard frontend, result data, request data, and any
evaluation backend. FlavourBench currently uses the two components it needs:

- `josefchen/flavourbench` dataset: immutable release tables, official task maps, and disjoint lab
  training maps;
- `josefchen/flavourbench` Space: a read-only explorer plus deterministic scoring endpoints.

The Space makes no provider calls. Endpoint keys and local checkpoints remain in the lab's
environment. The checked-in runner downloads the public task contract and emits response and
report artifacts that can be verified offline.

The Space carries the `leaderboard`, `modality:text`, `judge:auto`, and `submission:manual` tags so
it is discoverable through Hugging Face's leaderboard index. Native `eval.yaml` aggregation is a
separate integration: the FlavourBench evaluation framework must first be added to Hugging Face's
supported framework enum and benchmark allow-list. Do not publish an unsupported `eval.yaml`.

## Result publication

The Space scores an uploaded artifact in-session but never adds it to the leaderboard. A proposed
result must include all 534 responses, the content-addressed report, an immutable artifact URL,
route and decoding settings, and a training disclosure. The offline verifier must reproduce the
score. Accepted results enter a new release rather than changing an existing release.

See [submitting results](./submitting-results.md).

## Release gate

- [x] 27 by 534 response matrix is complete.
- [x] Dataset and lab manifests verify byte-for-byte.
- [x] Search, best-per-lab filter, download, and empty state work.
- [x] Desktop, mobile, light, dark, keyboard focus, and reduced motion are covered.
- [x] Space exposes the score and training APIs without accepting provider credentials.
- [x] Result submission template and verification contract are public in the source tree.
- [x] Dataset and Space remain private before the paper is approved.
- [ ] Make both Hugging Face repositories public when the approved arXiv record is live.
- [ ] Replace repository paper links with the arXiv URL after approval.
- [ ] Apply for Hugging Face native benchmark registration after public release.
