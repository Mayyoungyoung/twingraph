# Two-family value dataset v2

All records are from physical MuJoCo execution with the existing Panda feedback
controllers. This is simulation data, not real-robot data. Original v1 archives
and their `manifest.json` remain unchanged.

## Files

| File | Contents |
|---|---|
| `two_family_v2.tar.gz` | 74 complete decision groups: immutable inputs, complete candidate PlanIRs, paired physical outcomes, geometry features and visual cache |
| `manifest_v2.json` | Per-group split, trials, source/input hashes, collection work and every archived file SHA-256 |
| `two_family_v2_source.tar.gz` | Exact source, tests and all simulation assets used to reproduce this release |
| `source_hashes_v2.json` | Exact bytes of the source archive; runtime freeze commit and later reporting-script revisions are distinguished |
| `online_execution_v2.tar.gz` | Actual main, budget, scale and fresh-configuration decisions, inputs, complete plans and execution traces |
| `development_evidence_v2.tar.gz` | Pilot failures, physical parameter checks, order/spacing probes, regression and preserved wide-port development records |
| `wide_v2a_source.tar.gz` | Earlier wide-port collection source; explicitly admitted by the sampling-only compatibility manifest |

Release checksums are in
[`docs/evidence/value_v2/release_checksums.json`](../../docs/evidence/value_v2/release_checksums.json).
Its paths are relative to the server release directory: e.g. `manifest.json`
there is stored here as `manifest_v2.json`. Model paths in that release are
mapped to `models/value/v2/`. Individual JSON line endings may change in a Git
checkout; authoritative file hashes refer to the unchanged tar archive bytes.

## Split and label meaning

The 74 decision groups contain 57 distinct parameterized geometries and 44
family/seed clusters. All checkpoints from one seed, including wide/compact
variants, remain in one split. There are 44 training groups, 14 validation
groups, and 16 locked reference groups. Do not randomly split candidate rows.

There are 2,368 label rollouts: 1,856 development and 512 independent reference
trials. Each group has 16 candidates and two paired perturbations per candidate.
Full-program success occurs 1,484 times; every initial prefix succeeds. Rates
from two repeats are coarse observations, not ground-truth probabilities.
Program exceptions are excluded from physical failure labels and preserved as
development diagnostics. All-success/mixed/all-failure groups are retained;
this dataset happens to contain two all-success and no all-failure groups.

The main task distributions were fixed before opening locked labels. These test
seeds are now public and examined; a new method-development round needs a new
predeclared test split. Use the present test set for regression.

## Restore

From the repository root, in the existing server environment:

```bash
mkdir -p results/value_v2/data
tar -xzf datasets/value/two_family_v2.tar.gz -C results/value_v2/data
tar -xzf datasets/value/online_execution_v2.tar.gz -C results/value_v2
```

To train exactly on this release, restore the archive first. Regenerating only
with the current sampler does not recreate the retained early wide-port data.
See [run commands](../../docs/value-v2-running.md),
[full report](../../NEXT_ROUND_REPORT.md), and
[per-decision table](../../EXPERIMENT_TABLE.csv).

Collection, training and reference costs are reported separately. Offline cache
lookup is never used as the online planning timing measurement. All online
policies execute their chosen PlanIR again under the deployment perturbation
domain, independently of online validation and reference domains.
