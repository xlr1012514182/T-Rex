# Attribution and component terms

T-Rex × Revo 3 builds on the upstream **T-Rex: Tactile-Reactive Dexterous Manipulation** codebase. The root [LICENSE](LICENSE) and the original citation in [README](README.md#citation) are retained. The Revo integration should be identified separately from upstream architecture and reported results.

## Components

- `qwen_vla/`, `tactile_vqvae/`, `dataset_quickstart/` and `hardware_code/` retain upstream source and attribution. The dataset and bimanual hardware guides describe the upstream embodiment, not Revo experiment results.
- `teleop_data_collection/` declares Apache-2.0 in its package metadata. This root notice does not replace its declared terms or grant rights in any external SDK.
- Third-party source and robot assets under `hardware_code/third_party/` and `hardware_code/vive_tracker/` retain their included license, copyright and attribution files.
- External hardware sources are pinned in [sources.lock.json](teleop_data_collection/sources.lock.json) and [tianji_sources.lock.json](teleop_data_collection/tianji_sources.lock.json). See the [data-collection guide](teleop_data_collection/README.md#2-固定来源用途与许可证边界) before fetching or redistributing them.
- Qwen, GNI and other model weights, remote datasets, device SDKs and manufacturer binaries are governed by their own terms. A reference, download helper or adapter is not a sublicense of those artifacts.

Do not remove component notices when redistributing this repository or extracting a submodule. Check the terms of the exact external artifact version being used.
