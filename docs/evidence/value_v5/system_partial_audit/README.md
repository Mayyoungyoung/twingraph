# Completed Top-K cases 61400 and 61403: partial audit

Read-only JSON/source audit; no rollouts, rendering, inference, test suite, or aggregate system estimate. See `audit.json` for raw-file hashes and independently recomputed selection checks.

Frozen mlp_29 checkpoint, model selection, and classification thresholds are unchanged. All provenance, ranking, execution binding, namespace and selection checks pass. Each case actually used 8 twin rollouts (4 candidates x 2) and 1 independent target rollout.

- **61400:** candidate `3b03503d41790c66a955` (original pool index 9; model rank 4) uniquely won with twin 2/2. Target failed at `press_seat(pin_right)`, recorded handler index 76, with height error **46.54 mm** and contact force **1.413 N**. The preceding guarded descent correctly stopped on contact at z=0.901532 m; its success did not certify seating at target z=0.855 m. This is a valid uncensored physical failure; it does not conflict with the two twin successes.
- **61403:** candidate `5b37a469006c0a85a1d9` (original pool index 0; model rank 4) tied candidate at pool index 4 with twin 2/2 and won by original source order. The independent target passed all five final seating/release goals; maximum position error was **0.40443 mm**.

Both results record distinct Session/context/MjModel/MjData plus six disjoint important buffers, a separate target scene factory call, and no snapshot transfer. Frozen source enforces these checks, creates the target from scratch, rebinds its initial trajectory, and executes against that target's own runner. Twin namespace is 5107; target namespace is 7901. Same nominal initial-state and graph hashes arise from deterministic same-seed scene recreation, not state copying.

The target is another MuJoCo instance under independent draws within the same known friction/gain ranges. Two successes are finite evidence, not a success guarantee. No hardware claim, isolated causal attribution to friction/gain, or aggregate success/timing conclusion follows from these two cases.
