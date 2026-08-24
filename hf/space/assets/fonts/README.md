# Bundled web fonts

The Space bundles small WOFF2 subsets of the same typefaces used in the FlavourBench launch artwork so that its first render is deterministic and does not depend on a third-party font CDN.

- `Lato-*.woff2` was derived from Lato by Łukasz Dziedzic and is distributed under the SIL Open Font License 1.1. See `LICENSE-LATO.txt`.
- `DejaVuSansMono-*.woff2` was derived from DejaVu Sans Mono and is distributed under the Bitstream Vera font licence. See `LICENSE-DEJAVU.txt`.

The subsets contain the printable ASCII range, non-breaking space, and middle dot. They were produced with fonttools `pyftsubset`; characters outside that set fall back to the user's system fonts.
