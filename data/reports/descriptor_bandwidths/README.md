# Gaussian descriptor bandwidth audit

These files are historical diagnostic snapshots measured on local DW4, LJ13 and LJ55 training
splits with seed 42. Their model and noise settings are embedded in each JSON report and do not
match every current baseline config under `configs/`. The reports predate the explicit
`model.architecture` field and used the historical EGNN generator.

Each main report uses 512 reference configurations, 512 disjoint real query configurations, and
512 samples from the initialized generator. The descriptor is
`sort({||x_i-x_j|| : i<j}) / sqrt(N*(N-1)/2)`. No test data are used. JSON reports include the
model and noise settings, distance quantiles, nearest-reference distances, per-query kernel mass,
effective reference counts, and coordinate field magnitudes. Field RMS uses the first 128
generated queries and all 512 references/negatives, excluding each query's self interaction.
Timings and GPU throughput were not benchmarked.

## Selection rule

For descriptor separation `d`, the Gaussian kernel is `exp(-d^2/(2*h^2))`. The magnitude of
its descriptor gradient is `d/h^2 * exp(-d^2/(2*h^2))`; for fixed nonzero `d`, this is maximized
at `h=d/sqrt(2)`. This motivates a broad component based on typical generator-to-data distance,
alongside `0.5`, `1`, and `2` times the real-data median. These are heuristics, not an optimal
KDE bandwidth estimator. A local component at twice the median nearest-reference distance is
also included if it is smaller than half the global median. This matters for DW4: its median
nearest-reference distance is only 0.074 despite its global median of 0.917.

The bandwidth lists recommended when these snapshots were recorded round the candidate values:

| System | Real distances: 5% / 50% / 95% | Initial distances: 5% / 50% / 95% | Bandwidth list |
| --- | --- | --- | --- |
| DW4 | 0.154 / 0.917 / 1.604 | 0.617 / 1.391 / 3.291 | 0.15, 0.45, 0.9, 1.8 |
| LJ13 | 0.060 / 0.135 / 0.421 | 0.271 / 0.718 / 1.246 | 0.07, 0.14, 0.28, 0.5 |
| LJ55 | 0.021 / 0.044 / 0.105 | 0.127 / 0.202 / 0.328 | 0.022, 0.044, 0.088, 0.15 |

## What the measurements establish

- LJ13: at `h=0.067559`, about 36% of initial generated queries have **every** reference
  kernel below `1e-6`; at `h=0.507586`, that fraction is zero and mean attraction kernel mass
  is about 0.404. A narrow real-data-only bandwidth misses much of the initial generator.
- LJ55: at `h=0.021989`, about 83% have every reference kernel below `1e-6`, and attraction
  coordinate RMS is about `2e-6`. At `h=0.143111`, no sampled query fails that threshold and
  attraction coordinate RMS is about 0.0294. Broad attraction is necessary at initialization.
- DW4: `h=0.458665` leaves about 5.7% of initial queries below the same threshold; the broad
  component `h=1.834659` covers all sampled queries. The additional local component targets
  differences within the real-data neighborhoods.
- Reusing the old LJ13 list `[0.5, 1.5, 3, 5]` in descriptor space would mostly oversmooth:
  using the reported median separation 0.135, pair kernel values at `h=1.5,3,5` are approximately
  0.996, 0.999, and 0.9996. Those components barely discriminate typical real configurations.

## Limits and next training checks

The reports establish distance scales and initial coverage for these data/model settings.
They do not show convergence, improved energy distributions, or optimal bandwidths. Reference
bank size affects nearest-neighbor distances and coverage. The additional `lj55_bank64.json`
report checks a smaller 64-reference bank: real-data median distance is 0.0503 and initial
generator-to-data median distance is 0.2013, supporting the same approximate bandwidth range.
Reproduce that snapshot size with `--samples 64`. The median/global distance scale is less
sensitive to bank size than local coverage.
Different seeds and a trained generator can also change coverage substantially.

Keep both fine and broad components initially. Select any later bandwidth annealing and `eta`
adjustments using held-out energy Wasserstein distance, pair-distance distributions, collision
fraction, and generated radius. Coordinate field magnitude changes with the descriptor Jacobian,
so retaining the old `eta` is a starting experiment, not an assertion that it is optimal.
Check structural observables too: a sorted pair-distance distribution loses connectivity and is
not a unique encoding of every configuration, even though it retains the pairwise energy.

Run
`uv run --no-sync python -m utils.check_bandwidths --config <config> --output <report.json>`
from the repository root to audit a current configuration. Results are directly comparable to a
recorded snapshot only when its embedded model, noise, sample count, and seed settings match.
