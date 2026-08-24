# Frozen Season 0 analysis source

This directory preserves the ten Python modules named by the active Season 0 analysis artifact:

`season0-automated-analysis-ab45eff77098a97fc05ef7ee5ca689b00724381e4bd8c6f7e4dd60c86fb61d97.json`.

`MANIFEST.sha256` matches the artifact's embedded implementation manifest byte for byte. The
snapshot exists because the production service continued to evolve after the retrospective
analysis was frozen. Paper rendering validates this snapshot instead of treating the current
service source as the historical implementation.

The snapshot was recovered from the content-addressed Season 0 research archive plus the five
analysis changes, one cost-accounting change, and one newly added completion-policy module that
preceded the v6 analysis freeze. No provider execution or model generation is involved.
