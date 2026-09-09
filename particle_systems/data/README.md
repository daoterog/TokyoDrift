# Local data

Prepared datasets and source downloads are local-only and intentionally excluded from Git.
Run `python -m particle_systems.prepare_data dw4`, `lj13`, or `lj55` after creating the project
environment. LJ55 is published as two source arrays; both are downloaded and sampled without
concatenating them in memory.

To prepare all supported datasets on a compute node, run or submit
`particle_systems/jobs/download_datasets.sh` after running
`particle_systems/jobs/download_env.sh`.
