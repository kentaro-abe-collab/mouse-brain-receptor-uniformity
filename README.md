# Mouse Brain Receptor Uniformity Analysis

This repository contains analysis scripts, output CSV files, and visualization figures used in a study investigating neurotransmitter receptor composition across mouse brain macro-regions.  
The analyses are based on single-cell RNA-seq data from the **Allen Brain Atlas (WMB-10Xv3, *-log2.h5ad)**.

---

## 📂 Repository Structure

mouse-brain-receptor-uniformity/
│
├── src/ # Python analysis scripts
│ └── mouse_brain_receptor_uniformity_analysis.py
│
├── data/ # Output CSV files
│ ├── region_receptor_subtype_ratio.csv
│ ├── logistic_region_coefficients_CB_removed_isocortex_baseline.csv
│ └── region_by_gene_expression_percentage.csv
│
├── figures/ # Visualization figures (PNG)
│ ├── Figure1_receptor_heatmap.png
│ └── Figure2_logit_coeff_boxplot_CB_removed.png
│
├── requirements.txt # Python dependencies
├── LICENSE # MIT license
└── README.md # Documentation


---

## 🧠 Overview

This project performs two complementary analyses using receptor-related genes from the mouse whole-brain single-cell atlas:

### **(A) Region-level expressing-cell fraction analysis**
For each macro-region:

1. Compute expressing-cell fractions  
   \[
   \text{fraction} = \frac{\#(expression > 0)}{\text{total cells in region}}
   \]
2. Convert fractions to percentages  
3. Normalize each region row so that receptor subtypes sum to 100%  
4. Visualize the receptor-composition matrix as a heatmap

Outputs:

- `region_by_gene_expression_percentage.csv`  
- `region_receptor_subtype_ratio.csv`  
- `Figure1_receptor_heatmap.png`

---

### **(B) Logistic regression across brain regions**  
(Excluding cerebellum; Isocortex used as baseline)

For each receptor gene:

Logistic model:
\[
\logit P(y_{ij}=1) = \beta_0 + \sum_k \beta_k I(\text{region}=R_k)
\]
where:

- \( y_{ij}=1 \) if gene expression > 0  
- `"cb"` (cerebellum) is excluded  
- `"isocortex"` = baseline  
- One coefficient per non-baseline region

Outputs:

- `logistic_region_coefficients_CB_removed_isocortex_baseline.csv`  
- `Figure2_logit_coeff_boxplot_CB_removed.png`

---

## 📊 Data Source

The analysis requires the following dataset:

- **Allen Brain Institute – Mouse Whole-Brain (WMB-10Xv3) single-cell RNA-seq dataset**  
  (`*-log2.h5ad` files)

Due to licensing restrictions, raw h5ad files cannot be included in this repository.

---

## ▶️ How to Run the Analysis

1. Place Allen Brain Atlas `*-log2.h5ad` files in a directory.
2. Edit the script configuration:

```python
H5AD_DIR = Path(r"your/path/to/h5ad_files")

3.Run the analysis script
Execute the following command in your terminal or PowerShell:

python src/mouse_brain_receptor_uniformity_analysis.py

4.Check the output results
The script will automatically generate CSV and PNG files under:
./WMB_receptor_uniformity_results/
You may manually move the CSV/PNG files into /data/ and /figures/ for publication.

📦 Dependencies
Install required libraries via:

bash
pip install -r requirements.txt
Dependencies:

numpy

pandas

scanpy

statsmodels

matplotlib

seaborn

🔁 Reproducibility Notes
The script automatically detects region labels from the h5ad metadata.

Output directory is created reproducibly under the current working directory.

Logistic regression uses Isocortex as a fixed baseline across all genes.

📄 License
This project is distributed under the MIT License, allowing reuse with attribution.

📚 Citation
If you use this repository in academic work, please cite:

Mouse Brain Receptor Uniformity Analysis.
GitHub: https://github.com/yourname/mouse-brain-receptor-uniformity

(Replace with Zenodo DOI after deposit.)
