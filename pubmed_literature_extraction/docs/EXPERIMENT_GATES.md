# Agent v3 experiment gates

This project does not run five-fold evaluation by default.  The next
evaluation must pass the following sequential gates before any 100-article or
cross-fitting run is started.

1. Compatibility gate: legacy output must match the frozen baseline when
   `RELATION_AUTHORITY=legacy`; `unified-shadow` may add trace data only.
2. Budget/cache gate: Judge and Recovery requests must appear in
   `phases.agent_v2.remote_usage`; a warm replay must make no auxiliary remote
   request for an identical state.
3. Leakage gate: prompts, rules, few-shot examples and recovery guidance must
   contain no held-out PMID, source hash, gold relation, or gold endpoint.
4. Evidence gate: every selected span must be a contiguous source substring;
   adjacent-sentence windows must remain inside one section and paragraph.
5. Quality sentinel: run a small frozen cohort before broad evaluation.  Stop
   if semantic F1 is below the legacy baseline, strict import-ready precision
   drops, dangerous writes are nonzero, or remote zero-change rate exceeds
   20%.

Only after those gates pass should the project resume the deferred five-fold
design.  The fold runner must construct any few-shot/recovery examples from
the fold's induction partition and save the partition PMID hashes alongside
the frozen candidate manifest.
