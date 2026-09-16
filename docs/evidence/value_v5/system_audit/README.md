# Final paired system audit

Independent read-only recomputation from all 8 completed policy records. No simulation, replay, inference, retuning, or reranking. The earlier partial audit is unchanged. See `audit.json` for all raw-file hashes, per-case checks, selection reconstruction, shared-trial comparisons and phase timings.

All checks pass. Frozen mlp_29, original model selection and classification thresholds are unchanged. All four pairs have exactly identical 12-candidate inputs. TopK physically validates 4 candidates x 2 trials per case (32 total); full validates 12 x 2 (96 total). Together there are **128 actual twin calls and 8 independent target calls**, with no missing result or timeout. All **32 shared twin candidate/trial pairs** have exactly identical physical labels, errors, final positions, executed parameters and step counts; their own recorded wall times and trace paths belong to separate runs.

| Case | TopK target | Full target | Same selected candidate |
|---|---|---|---|
|61400|failure|success|no|
|61401|success|success|yes|
|61402|success|failure|no|
|61403|success|success|yes|

Both policies achieve **3/4**, on different cases. Four paired cases and one target draw each cannot establish noninferiority or equivalence; the published success-difference bootstrap interval is [-0.75, 0.75].

TopK61400 fails at `press_seat(pin_right)`, 46.54 mm above target after contact-stop. Full61402 fails the final `inspect_seat(pin_left)`: position error **1.799484 mm** exceeds the fixed **1.5 mm** tolerance (tilt1.41167 degrees). Both are valid uncensored physical negatives, despite the selected candidate passing2/2 twin trials. All eight selections obey empirical twin success fraction then original pool order.

Every target starts with all five parts unassembled, owns separately checked simulator objects and buffers, receives no twin snapshot, and executes the selected semantic program after rebinding only its initial trajectory. Twin namespace5107 and target7901 are distinct; paired policies use the same role-specific disturbance draws. Identical nominal initial hashes are expected from deterministic same-seed scene reconstruction.

| Mean seconds over 4 paired cases | TopK | Full |
|---|---:|---:|
|Resident decision|338.569214|782.015804|
|Resident scene-to-terminal total|387.367979|834.936922|
|Model-cold decision|338.580220|782.015804|
|Model-cold total|387.378984|834.936922|

The **mean of per-pair decision ratios is2.339403**, and the **mean of total ratios is2.179592**. The ratio of decision means is2.309766; these aggregation definitions should not be interchanged. They are measured ratios, not the nominal budget ratio3. Mean model initialization is0.011006 seconds; Torch import is separately recorded (mean1.166914 seconds). Actual parallel batch wall is1352.247747 seconds. Decision sums candidate generation, graph/integrity/scoring, actual validation and selection; total additionally covers each own scene, target execution and terminal render. Offline LLM proxy generation, training, process startup and Torch import are excluded.

TopK always preceded full in each of four parallel workers; timing reflects this declared scheduling and changing resource contention, without counterbalanced order. The target remains another MuJoCo instance under known nuisance bounds. These observations do not establish hardware transfer, long-term assembly quality or sliding-stroke functionality.
