# Report experiment archive

This directory contains the raw data used by `report/report.tex`. The archive
is intentionally separate from the derived figures and statistics.

## Recorded trials

Both trials are based on Git commit
`58b32939cd4030ec9d2aec86003926bc575a4b59` and use a 30 ms simulation step.

| Profile | Start time (Europe/Berlin) | Duration | Samples |
| --- | --- | ---: | ---: |
| `stretch_esdf_offline_ompl_wbmpc` | 2026-08-14 12:25:30 | 40.02 s | 1,334 |
| `ur10_esdf_offline_ompl_wbmpc` | 2026-08-14 12:51:35 | 60.00 s | 2,000 |

Each timestamped `raw/` directory contains the unmodified `data.npz` written
by the experiment runner and its fully expanded `config.yaml`. The UR10
archive records `controller.acados.cython.recompile: true`: solver generation
was enabled because the clean workspace did not contain the configured
pre-generated UR10 acados JSON. No controller parameter was otherwise changed.

SHA-256 checksums:

```text
48556b2f89123df82d7f2519bc08eddf60d5e9812194324e6daa9fc0803dc169  data.npz
4d1f2d04dbbd19aacef97a3ded20ecd76c8d0340dc04586a323601e637dc4b0f  config.yaml
b67af19edba4d9cf3cd48228e1035f2152dd7ed809613f7e490cc00abe51c23c  ur10 data.npz
0bef5bf39175375ffc033a6d6d13dc972b263770b5bde242df097257f98e8647  ur10 config.yaml
```

Regenerate the metrics and PDF figures from the repository root:

```bash
pixi run python report/analyze_experiment.py \
  results/report_experiments/stretch_esdf_offline_ompl_wbmpc/2026-08-14_12-25-30/raw/data.npz \
  --ur10-data results/report_experiments/ur10_esdf_offline_ompl_wbmpc/2026-08-14_12-51-35/raw/data.npz \
  --output report/figures
```

This writes `report/experiment_metrics.json` and the figures used by the
LaTeX report. The archive contains one trial per robot; it should not be
interpreted as a multi-seed benchmark or a controlled comparison of robot
hardware.
