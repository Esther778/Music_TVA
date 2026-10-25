# AI Collaboration Guide

Last updated: 2026-06-23

This document should be read at the start of a new AI/Codex conversation for
this project.

## Working Style

- Complete the user's requested task first.
- After the requested task is handled, add independent technical judgment when
  it can improve the research, prevent weak claims, or reveal a better path.
- Do not limit discussion only to the exact command or experiment requested by
  the user. The user expects the AI to contribute research judgment, not just
  execute instructions.
- Keep suggestions grounded in current project data, scripts, metrics, and
  paper-writing constraints.
- Distinguish clearly between confirmed results, likely interpretations, and
  speculative next steps.

## Communication Rules

- Before doing work, say what will be done next, why it matters, and which files
  may change.
- During long-running work, give concise progress updates.
- When an experiment finishes, report the practical conclusion first, then the
  supporting metrics.
- Every new model report must compare against the strongest previous models.
  Include a same-protocol comparison as the primary evidence and a separately
  labeled historical comparison for older or non-comparable protocols.
- Every VA model report must include a threshold gap table against minimum
  pass, good, and excellent targets. Do not report only the current values; show
  how far each metric is from each target. See
  `research/MODEL_METRIC_REPORTING_STANDARD.md`.
- For VA models, report Valence and Arousal separately. A new aggregate score
  must not hide a regression in either dimension.
- If a result looks good but has a methodological caveat, state the caveat
  directly.
- For research claims, avoid overstating model superiority unless the data split,
  evaluation protocol, and metrics are comparable.

## Documentation Rules

- Keep project records updated so a new AI can quickly know:
  - what the project is doing,
  - what has already been tried,
  - which results are reliable,
  - what remains unresolved,
  - what should be done next.
- Update the most specific relevant document first, then update the top-level
  or cross-phase log if the decision affects the broader project.
- Do not overwrite published baseline artifacts unless explicitly requested.
- Preserve generated result files, checkpoints, metrics, and logs for later
  reporting.

## Research Judgment Rules

- Treat the published OpenL3+MLP result as a real, reproducible prior result.
- Keep using PyTorch for model training and verification. TensorFlow may appear
  only as an OpenL3 feature-extraction dependency.
- Remember the critical comparison caveat:
  - Published/OpenL3 MLP results use frame-level random splitting.
  - Current MERT results use song-level splitting.
  - These are not directly comparable as a fair feature-backbone comparison.
- For the EI paper, the stronger direction is section-aware complete-song
  emotional structure analysis, not simply claiming a better VA predictor.

## Scope Discipline

- Keep file edits scoped to the current research or QC task.
- Do not mix unrelated dirty files into a result.
- Do not run production Netlify deployment unless explicitly asked.
- Generated data and model outputs can be used locally, but should remain
  gitignored unless the user explicitly asks otherwise.
