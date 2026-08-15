# Forschungspraxis report

The completed report is maintained in `report.tex`; the compiled handoff is
`report.pdf`. It is written as an IEEE-style technical paper and currently
contains:

- the configurable whole-body model and platform boundary;
- nonholonomic 8-DoF Stretch and holonomic 9-DoF mobile-UR10 specializations;
- OMPL and offline-ESDF planning/control integration;
- the 8-D/11-D mimic-joint and real-command mapping;
- the guarded ROS 2 command-adapter design;
- measured two-platform simulation tracking, clearance, solver, and timing results;
- clearly separated real-interface validation evidence and limitations; and
- a reproducibility appendix and bibliography.

## Rebuild the experiment figures

From the repository root:

```bash
pixi run python report/analyze_experiment.py \
  results/report_experiments/stretch_esdf_offline_ompl_wbmpc/2026-08-14_12-25-30/raw/data.npz \
  --ur10-data results/report_experiments/ur10_esdf_offline_ompl_wbmpc/2026-08-14_12-51-35/raw/data.npz \
  --output report/figures
```

The script writes `report/experiment_metrics.json` and the PDF figures used
by the paper. It reads only the archived NPZ, so figure regeneration does not
depend on importing ROS, the simulator, or the controller.

## Rebuild the PDF

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

The report uses an IEEE two-column layout without imposing a fixed page limit.
Author/affiliation details, the scope of real-robot claims, and the decision to
present one trial per robot are the main items to revisit during manual editing.

## Archived data

See `../results/report_experiments/README.md`. Both raw NPZ files and expanded YAML
are tracked under `results/report_experiments/`; derived statistics and plots
remain under `report/`.
