# Independent locked-test audit

Selected model: `mlp_29`; frozen threshold `0.5695365870155553`. All integrity checks passed: True. Metric discrepancies: [].

- Nominal outcomes: 12 configurations, 144 candidates, 65 successful and 79 unsuccessful; no missing, all-failure or all-success groups.
- Confusion: TP36, FP12, TN67, FN29. Accuracy 103/144 = 71.5278%; balanced accuracy 70.0974%; precision 75%; recall 55.3846%; Brier 0.1932802849.
- Hit@4: 11/12 = 91.6667%; exact uniform-random expectation 84.2256%; source order also 11/12.
- Top4 nominal successes: 32/48 = 66.6667%; source order 22/48 = 45.8333%; random expected quality 45.1389%.
- Input diagnostics: unseen fields, changed training constants and broken duplicate relations each affect 0/144 candidates.
- Disjoint configuration counts: train 48, validation 12, test 12. Checkpoint, freeze, dispatch, prediction and raw-label bindings agree.

These are R=1 nominal simulation results on twelve configurations. Screening increases selected success density; Hit@4 does not exceed source order. Full-twin runtime and independent target success require the separately measured system experiment.

The accompanying audit.json preserves independent point recomputations, per-configuration results, official configuration-bootstrap intervals, hashes, chronology and limitations. No model, threshold, raw outcome or candidate order was changed.

Hash checks use the physical/dispatch producer's ordinary JSON separators and the evaluator's compact separators separately; those object digests are intentionally distinct from file-byte SHA256s.
