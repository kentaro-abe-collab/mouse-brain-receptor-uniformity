# Mouse Brain Receptor Uniformity Analysis

This repository contains **all code, processed data tables, and figure-generation outputs**
used in the manuscript:

> **Cross-regional uniformity of neurotransmitter receptor transcript detection across the adult mouse brain**

The analysis is based on publicly available single-cell RNA-seq data from the
**Allen Institute Adult Mouse Brain Cell Atlas (Cell Census v2023; WMB-10Xv3)**.

---

## Overview

**Cross-regional uniformity of neurotransmitter receptor transcript detection**  
*A quantitative framework for cross-regional uniformity*

This repository is a **methods-first, code-centric research artifact**.

The primary contribution is a **fully reproducible quantitative framework** for evaluating
cross-regional uniformity in **compositional molecular data**, with explicit uncertainty
estimation.

The accompanying manuscript (PDF) is provided as a **descriptive companion** to the
executable analysis pipeline; the authoritative research output is the code itself.

This repository provides a fully reproducible analysis pipeline for quantifying the degree of
cross-regional uniformity in neurotransmitter receptor–related transcript detection across
major macro-regions of the adult mouse brain.

Although the motivating application is biological, the **core contribution is methodological**:
a general framework for assessing *uniformity versus variation* in compositional data.

Specifically, the analysis integrates:

- **Compositional data analysis** (CLR transform, Aitchison geometry)
- **Absolute percentage-point difference metrics** (U75, U95)
- **Hierarchical bootstrap** for uncertainty quantification
- **Explicit region–pair–level summary statistics**

The framework is **not specific to particular genes or brain regions** and is extensible to
other molecular systems, brain structures, and cross-species comparisons.

---

## 📂 Repository Structure (Canonical)

```
mouse-brain-receptor-uniformity/
│
├── src/                      # Analysis code
│   └── Mouse_brain_receptor_uniformity_pipeline.py
│
├── data/                     # All non-figure outputs (CSV / JSON / TXT)
│   ├── region_detection_fractions.csv
│   ├── region_cell_counts_primary.csv
│   ├── region_compositions.csv
│   ├── uniformity_U75_U95.csv
│   ├── aitchison_distance_matrix.csv
│   ├── bootstrap_summary.csv
│   ├── bootstrap_distributions.csv
│   ├── run_metadata.json
│   └── analysis_log.txt
│
├── figures/                  # Publication-ready figures (PNG only)
│   ├── Figure1_heatmap.png
│   ├── Figure2A_aitchison_heatmap.png
│   └── Figure2B_aitchison_hist.png
│
├── manuscript/               # Manuscript PDF (descriptive companion)
│   └── cross_regional_uniformity_neurotransmitter_receptor_transcript_detection_mouse_brain.pdf
│
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # This document

```

**Rule enforced**  
- `figures/` → **PNG files only** (exactly those referenced as figures in the manuscript)  
- `data/` → **everything else** (CSV, JSON, TXT, logs, matrices, bootstrap outputs)

This structure matches:
- the *Code and Data Availability* section in the manuscript
- the actual outputs produced by the pipeline
- arXiv reproducibility expectations

---

## 🧠 Analytical Framework

The pipeline quantifies **cross-regional similarity (uniformity)** of neurotransmitter receptor–related transcript detection across **11 non‑cerebellar brain macro‑regions**.

Two complementary metrics are used:

1. **Gene-wise absolute percentage-point differences** (U75 / U95)
2. **Compositional distances** in CLR space (Aitchison distance)

Uncertainty is quantified using a **hierarchical bootstrap** (donor → cell).

---

## 📊 Primary Outputs

### Region-level detection and composition

| File | Description |
|----|----|
| `region_detection_fractions.csv` | Fraction of cells with detected expression (X > 0) per region × gene |
| `region_cell_counts_primary.csv` | Total number of cells per macro-region |
| `region_compositions.csv` | Row-normalized receptor subtype compositions (sum = 1 per region) |

### Uniformity metrics

| File | Description |
|----|----|
| `uniformity_U75_U95.csv` | Gene-wise upper-quantile (75%, 95%) absolute differences (percentage points) |
| `aitchison_distance_matrix.csv` | 11×11 matrix of pairwise Aitchison distances |

---

## 📈 Figures (Reproducible)

| Figure | File |
|----|----|
| Figure 1 | `Figure1_heatmap.png` — receptor subtype compositions across regions |
| Figure 2A | `Figure2A_aitchison_heatmap.png` — cross-regional Aitchison distances |
| Figure 2B | `Figure2B_aitchison_hist.png` — distribution of pairwise distances |

All figures are generated directly by the pipeline with fixed parameters.

---

## 🔁 Hierarchical Bootstrap Outputs

Bootstrap configuration (default):
- **1000 replicates**
- Sampling hierarchy: **donor → cell**
- Cell resampling implemented via **Binomial(n, p̂)** equivalence for Bernoulli detection

| File | Description |
|----|----|
| `bootstrap_summary.csv` | Point estimates with 95% confidence intervals |
| `bootstrap_distributions.csv` | Replicate-level bootstrap values |

---

## 🧪 Software Environment

All analyses were developed and executed under the following environment:

- **Python**: 3.12
- **NumPy**: 2.3.5
- **Pandas**: 2.3.3
- **Scanpy**: 1.11.5
- **Statsmodels**: 0.14.5
- **Matplotlib**: 3.10.7
- **Seaborn**: 0.13.2

This configuration ensures full reproducibility of the expressing-fraction calculations,
regional aggregation, compositional analyses, and hierarchical bootstrap procedures
reported in the manuscript.

---

## ▶️ How to Run

1. Prepare Allen Brain Atlas `*-log2.h5ad` files locally
2. Edit the input path in the script:

```python
H5AD_DIR = Path("/path/to/WMB-10Xv3/log2_h5ad")
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the pipeline:

```bash
python src/Mouse_brain_receptor_uniformity_pipeline.py
```

All outputs will be written to:

```
WMB_receptor_uniformity_results/
```

In this repository, the files corresponding to the manuscript are already organized
under `data/` and `figures/` as described above.

---

## 🧾 Reproducibility Notes

- Detection rule: **log2(expression) > 0**, equivalent to **UMI > 0** for WMB‑10Xv3 data
- CLR transform uses a fixed pseudo‑count: δ = 1e‑6
- Random seed is fixed and recorded in `run_metadata.json`
- Software versions and parameters are logged in `analysis_log.txt`

---

## 📄 License

MIT License — free reuse with attribution.

---

## 📚 Citation

If you use this code or data, please cite the Zenodo archive (canonical reference):

doi:10.5281/zenodo.18268080

GitHub repository:
https://github.com/sotomitiouru-collab/mouse-brain-receptor-uniformity

