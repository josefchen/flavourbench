# Anonymous ICLR 2027 paper source

Build and verify the manuscript with:

```bash
make
```

The command creates `main.pdf` with a fixed source date, checks the 19-page compiled package,
US-Letter geometry, embedded fonts, unresolved references, and box overflows. The counted main
text ends on page 8; the mandatory AI-use statement begins on page 9, followed by references and
appendices.

The unmodified official ICLR 2027 style archive was retrieved from:

<https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip>

Archive SHA-256:
`0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`

Relevant vendored-file SHA-256 values are recorded by the outer archive manifest. Leave
`\iclrfinalcopy` commented during double-blind review.
