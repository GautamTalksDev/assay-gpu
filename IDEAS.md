# IDEAS.md

Feature ideas that were **not** built. Do not open these. Correctness
bugs only. If an idea looks like a product, it stays here.

1. Ship `assay watch` after a GPU tokens/sec measurement that meets
   the original KT-2 10% bar. Do not widen the bar to 25%.
2. Attention-module ABFT with a noisefloor-v1 characterization path
   (today watch records MultiheadAttention as measurement-only
   INCONCLUSIVE).
3. An Assay issuance service that can actually emit `assay-signed`
   reports. v1 refuses.
4. A default `--every N` for watch derived from measured tokens/sec
   traces so overhead stays under 5%. No such traces exist; do not
   guess N.
5. Provider API wrappers that destroy the instance when `assay run`
   exits. Convenience is how a leaked token stays billed; destroy in
   the console and set a phone timer.
6. A public dashboard / heatmap of survey GPU models and verdicts.
7. Naming a rental provider after three reproductions. The budget
   does not buy three. Aggregate statistics only.
8. A `transformers` extra so watch can download an open-weights
   model. Watch makes zero network requests; do not add a downloader.
9. Burning both GCP $300 and Azure $200 new-user credits on this
   survey. Use one hyperscaler credit, not both.
10. Launch blog, tweet, Show HN, or "week nine" announcement. Six
    weeks of evidence. Do not post.
11. First-class ROCm / AMD rows in the survey harness.
12. Slack, email, or webhook on FAIL. That is telemetry. `assay run`
    makes zero network requests.
13. A cost predictor that guesses instance-hour price before rent.
    Preflight knows spend-to-date, not the next invoice.
14. A multi-provider orchestrator that rents 12 boxes in parallel.
    Breadth is one cheap instance at a time, logged before the next.
15. A network-egress monitor sidecar as a productized CI check.
16. Auto-characterize noisefloor on first GPU contact inside `assay
    run`. Characterization is a separate, logged spend.
17. A web UI for browsing attestation-v1 JSON.
18. Re-tuning KT-1 thresholds because the detection matrix is empty.
    The detector is not retuned.
19. Cache `eᵀ |A| |B| e` across GEMMs. residual-v2's scale depends
    only on A and B. At inference, weights are fixed across calls, so
    the `|B|` factor could be cached per weight matrix and amortized.
    Untested. Do not build this to paper over T4 overhead at 2048³.

When one of these becomes necessary, copy it into a checkpoint and
delete it from this list. Until then, leave it.
