# Checkpoints

[`manifest.json`](manifest.json) records four original main-model weight files, their
source filenames, sizes, SHA256 checksums and model entrypoints. All weight
binaries are versioned with Git LFS; the manifest and checksums remain regular Git files.

| Path | Model | Purpose |
|---|---|---|
| `pretrained/imu.pt` | `models.imu.TemporalConvEncoder` + `models.unimodal.UniModalityModel` | Original IMU initialization |
| `pretrained/kp.pt` | `models.skeleton.STGCN_Encoder` + `models.unimodal.UniModalityModel` | Original keypoint initialization |
| `ramoe/best.pt` | `models.ramoe.PRISM` | Main historical `best` file |
| `ramoe/latest.pt` | `models.ramoe.PRISM` | Main historical `latest` file |

`best` and `latest` are the original filenames' labels. Historical training
participants and selection details have not been fully reconstructed; checksums
do not establish accuracy or an evaluation protocol. Separate raw-data runs and
later repaired-model, v2, S+JS and MoBind weights are not included.

Install Git LFS, then download and verify the files from the repository root:

```bash
git lfs install
git lfs pull
sha256sum -c checkpoints/SHA256SUMS
```

For an offline copy from the original local archive, replace `SOURCE_DIR` with
the directory containing the original exported filenames:

```bash
SOURCE_DIR=/path/to/local/legacy
mkdir -p checkpoints/pretrained checkpoints/ramoe
cp -n "$SOURCE_DIR/imu_pretrained.pt" checkpoints/pretrained/imu.pt
cp -n "$SOURCE_DIR/kp_pretrained.pt" checkpoints/pretrained/kp.pt
cp -n "$SOURCE_DIR/ramoe_best.pt" checkpoints/ramoe/best.pt
cp -n "$SOURCE_DIR/ramoe_latest.pt" checkpoints/ramoe/latest.pt
sha256sum -c checkpoints/SHA256SUMS
```

`cp -n` keeps existing files. All four checksum checks must report `OK` before
using the weights; a mismatch means the local file is a different checkpoint.
For the local WSL workspace, the original source directory is
`/mnt/d/Repository/RAMoE/checkpoints/legacy`. Runtime scripts use the canonical
paths above, with no legacy aliases or symbolic links.
