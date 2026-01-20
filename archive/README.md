# Archive: Exploratory Stability Experiments

This directory contains exploratory scripts used to calibrate the physical stability of soil_bin=1.

**Results consolidated into:** `experiments.ipynb`

## Files

- `stability_experiments.ipynb` - Initial exploratory notebook (partially executed)
- `stability_script.py` - Initial systematic parameter experiments (Baseline, A1-A2, B1-B2, D1-D2)
- `stability_aggressive.py` - Aggressive parameter experiments (E1-E5)
- `validate_stability.py` - Independent validation of E3 configuration
- `final_config_test.py` - Head-to-head comparison E3 vs E3+
- `STABILITY_REPORT.md` - Comprehensive experimental report

## Final Configuration

**E3+ Parameters (100% success rate):**
- `rain_range=(0.0, 0.8)`
- `max_soil_moisture=320.0`
- `et0_range=(2.0, 8.0)` (unchanged)
- Mean bin-1 residence: 11.28 steps (target: ≥10)

These parameters are now applied in `experiments.ipynb` by default.
