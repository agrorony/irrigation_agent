# Quick Reference: Navigation Guide

This guide helps you quickly navigate the irrigation agent project documentation.

---

## 📚 Documentation Structure

### Main Documents
1. **[README.md](README.md)** - Project overview, quick start, installation
2. **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - **[PRIMARY REFERENCE]** Comprehensive research status and roadmap
3. **QUICK_REFERENCE.md** (this file) - Navigation shortcuts

### Supporting Materials
- **[experiments.ipynb](experiments.ipynb)** - Full experimental workflow with results
- **[archive/STABILITY_REPORT.md](archive/STABILITY_REPORT.md)** - Detailed stability calibration experiments
- **[archive/README.md](archive/README.md)** - Archive of exploratory scripts

---

## 🎯 Quick Navigation by Question

### "What has been accomplished so far?"
→ **[PROJECT_STATUS.md § 1.1](PROJECT_STATUS.md#11-what-has-been-achieved)**
- Physical environment calibration ✅
- Q-learning convergence (dry regime) ✅
- Regime shift experiment (moderate rainfall) ✅

### "What can I trust about the results?"
→ **[PROJECT_STATUS.md § 1.2](PROJECT_STATUS.md#12-guarantees-and-validations)**
- Stability guarantees ✅
- Convergence guarantees ✅
- Interpretability guarantees ✅

### "What are the known limitations?"
→ **[PROJECT_STATUS.md § 1.3](PROJECT_STATUS.md#13-known-limitations)**
- State coverage (50% visited)
- Regime-specific assumptions
- State space design constraints

### "What questions do Q-tables answer?"
→ **[PROJECT_STATUS.md § 2.1](PROJECT_STATUS.md#21-what-the-q-table-already-answers)**
- Climate-specific optimal policies ✅
- Regime sensitivity analysis ✅
- Physical interpretability ✅
- Policy compactness ✅
- Action distributions ✅

### "What questions can Q-tables NOT answer?"
→ **[PROJECT_STATUS.md § 2.2](PROJECT_STATUS.md#22-what-the-q-table-cannot-answer-by-design)**
- Generalization outside training ❌
- Continuous state interpolation ❌
- Causal explanations ❌
- Transfer learning ❌
- Multi-objective optimization ❌
- Uncertainty quantification ❌
- Multi-step planning ❌

### "What should I do next?"
→ **[PROJECT_STATUS.md § 3](PROJECT_STATUS.md#3-proposed-next-steps-high-level-research-directions)**

**Phase 1 (Immediate - 1-2 weeks):**
1. Comparative policy analysis (state-by-state divergence mapping)
2. Baseline comparisons (Q-learning vs. heuristics)

**Phase 2 (Near-term - 2-4 weeks):**
3. Robustness checks (hyperparameter sensitivity)
4. Visualization suite (heatmaps, decision trees)

**Phase 3 (If time permits - 4+ weeks):**
5. Cross-regime transfer tests
6. Interactive dashboards
7. Full technical report

---

## 🔬 Research Use Cases

| Use Case | Primary Document | Key Sections |
|----------|------------------|--------------|
| **Understanding project status** | PROJECT_STATUS.md | § 1.1, § 1.2 |
| **Identifying limitations** | PROJECT_STATUS.md | § 1.3, § 2.2 |
| **Planning next experiments** | PROJECT_STATUS.md | § 3, § 4 |
| **Learning the methodology** | experiments.ipynb | All cells |
| **Understanding stability** | archive/STABILITY_REPORT.md | Experimental timeline |
| **Quick code examples** | README.md | Quick Start |

---

## 📊 Key Metrics Reference

### State Space
- **Total states:** 36 (12 soil bins × 3 crop stages × 2 ET₀ bins × 2 rain bins)
- **Action space:** 3 (0mm, 5mm, 15mm)
- **Q-table size:** 36 × 3 = 108 values

### Training Performance
- **Convergence:** ~500 episodes
- **Mean reward:** ~177 (both regimes)
- **State coverage:** 50% (18/36 states visited in dry regime)

### Physical Parameters
| Regime | Rain Range | Rain/ET Ratio | Policy Irrigation % |
|--------|------------|---------------|---------------------|
| Dry | 0-0.8 mm/day | 0.09 | 19.4% |
| Moderate | 0-3.0 mm/day | 0.33 | 11.1% |

### Stability Metrics
- **Soil bin 1 residence:** 11.28 steps (target: ≥10)
- **Validation success:** 100% (5/5 independent trials)
- **Parameter configuration:** rain_range=(0, 0.8), max_soil_moisture=320

---

## 🚀 Getting Started Paths

### For Researchers (Academic Analysis)
1. Read [PROJECT_STATUS.md § 1](PROJECT_STATUS.md#1-current-status-technical-summary) - Understand achievements
2. Read [PROJECT_STATUS.md § 2](PROJECT_STATUS.md#2-role-of-the-q-table-capabilities-and-boundaries) - Understand capabilities/limitations
3. Read [PROJECT_STATUS.md § 3](PROJECT_STATUS.md#3-proposed-next-steps-high-level-research-directions) - Plan next steps
4. Review [experiments.ipynb](experiments.ipynb) - See methodology in action

### For Developers (Code Implementation)
1. Read [README.md](README.md) - Quick start and installation
2. Study [irr_Qtable.py](irr_Qtable.py) - Core Q-learning implementation
3. Study [irrigation_env.py](irrigation_env.py) - Environment dynamics
4. Run [experiments.ipynb](experiments.ipynb) - Execute training

### For Reviewers (Quality Assurance)
1. Read [PROJECT_STATUS.md § 1.2](PROJECT_STATUS.md#12-guarantees-and-validations) - Validation guarantees
2. Read [PROJECT_STATUS.md § 1.3](PROJECT_STATUS.md#13-known-limitations) - Known limitations
3. Review [archive/STABILITY_REPORT.md](archive/STABILITY_REPORT.md) - Experimental rigor
4. Check [experiments.ipynb](experiments.ipynb) - Results reproducibility

---

## 📝 Document Lengths

| Document | Lines | Purpose |
|----------|-------|---------|
| PROJECT_STATUS.md | ~488 | Comprehensive research roadmap |
| README.md | ~109 | Project overview |
| QUICK_REFERENCE.md | ~150 | Navigation guide (this file) |
| experiments.ipynb | - | Interactive experimental workflow |
| archive/STABILITY_REPORT.md | ~196 | Stability calibration details |

---

## ⚠️ Important Notes

### What This Project Is:
✅ Tabular Q-learning for **interpretable policy analysis**  
✅ Climate-adaptive irrigation strategy **research tool**  
✅ Academic demonstration of **regime sensitivity**  

### What This Project Is NOT:
❌ Production irrigation controller  
❌ Deep reinforcement learning (no neural networks)  
❌ Real-time decision system  
❌ Multi-objective optimizer  

---

## 🔗 External References

- **Q-Learning Algorithm:** Watkins & Dayan (1992)
- **Gymnasium Framework:** https://gymnasium.farama.org/
- **State Discretization:** Classic tabular RL approach
- **Climate Parameters:** ET₀ (Penman-Monteith), rainfall (stochastic sampling)

---

*Last Updated: January 2026*  
*For questions or clarifications, refer to [PROJECT_STATUS.md](PROJECT_STATUS.md)*
