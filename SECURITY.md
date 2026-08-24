# Security policy

## Reporting

Please report a suspected credential leak, authorization bypass, unsafe benchmark execution path,
or privacy issue privately through GitHub Security Advisories for this repository. Do not open a
public issue containing secrets, personal data, or exploit details.

## Public-repository boundary

This repository contains public benchmark source, tests, manuscript assets, and bounded research
records. It must not contain:

- provider keys or tokens;
- `.env` files or private route configuration;
- participant identity or contact data;
- live databases, private evidence archives, or production logs;
- unrestricted Epicure data payloads; or
- credentials embedded in notebooks, fixtures, issue text, or commit history.

Test credentials are syntactically fake and must remain unmistakably marked as non-real.

## Supported versions

The default branch is the only supported public research snapshot until versioned releases begin.
