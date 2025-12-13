# mouse_brain_receptor_uniformity_analysis.py
"""
Mouse brain receptor uniformity analysis
---------------------------------------

This script performs a two-step analysis on the Allen Brain Atlas mouse
single-cell RNA-seq data (WMB-10Xv3 *-log2.h5ad files):

(A) Region-level expressing-cell fraction and heatmap
    - For each *-log2.h5ad file, and for each macro brain region:
          expressing_fraction = (# cells with expression > 0) / (total # cells in region)
      for each receptor gene.
    - Fractions from multiple files sharing the same region name are averaged.
    - Expressing fractions are then converted to percentages:
          fraction (0–1) × 100 → percentage (0–100), rounded to 3 decimal places.
    - Percentages are normalized per region so that each row sums to ~100%,
      yielding a receptor-composition profile for each macro region.
    - The result is saved as:
          region_by_gene_expression_percentage.csv
      and visualized as a heatmap:
          Figure1_receptor_heatmap.png

(B) Cell-level logistic regression and boxplot (CB removed, Isocortex baseline)
    - For each receptor gene, we pool all cells across files and fit a
      region-wise logistic regression on binary expression:
          y_ij = 1 if expression_ij > 0 else 0
          logit P(y_ij = 1) = β0 + Σ_k β_k * I(region_ij = R_k)
      where:
          - Regions with label "cb" (cerebellum) are excluded a priori.
          - "isocortex" is used as the reference (baseline) region.
          - One dummy variable per non-baseline region (macro division).
    - For each gene, we collect the set of region coefficients {β_k} across
      non-baseline regions and summarise their distribution as a boxplot.
      These coefficients represent the log-odds difference in expression
      relative to isocortex.
    - The coefficients are saved as:
          logistic_region_coefficients_CB_removed_isocortex_baseline.csv
      and visualised as:
          Figure2_logit_coeff_boxplot_CB_removed.png

All outputs are written under:
    ./WMB_receptor_uniformity_results/

Usage:
    python mouse_brain_receptor_uniformity_analysis.py
"""

import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Directory containing WMB-10Xv3 *-log2.h5ad files
H5AD_DIR = Path(r"C:\Users\abeke\abc_cache\expression_matrices\WMB-10Xv3\20230630")

# Output directory: under the current working directory (reproducible and
# easy for reviewers to locate).
OUT_DIR = Path.cwd() / "WMB_receptor_uniformity_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "density_log.txt"
if LOG_FILE.exists():
    LOG_FILE.unlink()  # Remove previous log file

# Target receptor genes (neurotransmitter-related receptors)
CANDIDATE_RECEPTOR_GENES = [
    "Gabra1", "Gabra2", "Gabra3", "Gabrb1", "Gabrb2", "Gabrg2",
    "Htr1a", "Htr2a", "Htr2c", "Htr7",
    "Drd1", "Drd2", "Drd3", "Drd4",
    "Oprk1", "Oprm1", "Oprd1",
    "Grin1", "Grin2a", "Grin2b", "Gria1", "Gria2",
    "Adra2a", "Adra2c", "Chrna4", "Chrnb2",
]

# Possible region-label columns in the h5ad metadata
REGION_KEY_CANDIDATES = [
    "region_label",
    "anatomical_region_label",
    "anatomical_division_label",
    "struct_acronym",
    "structure",
    "structure_acronym",
    "structure_label",
]

LOG2_PATTERN = "-log2.h5ad"

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def find_log2_files(h5ad_dir: Path):
    """Return all *-log2.h5ad files in the specified directory."""
    return sorted([p for p in h5ad_dir.iterdir() if LOG2_PATTERN in p.name])


def detect_region_column(obs_columns):
    """Automatically detect the best region column based on known candidates."""
    for col in REGION_KEY_CANDIDATES:
        if col in obs_columns:
            return col
    return None


def normalize_region_series(s):
    """Normalize region labels: strip spaces and convert to lowercase."""
    return s.astype(str).str.strip().str.lower()


def compute_expressing_fraction(h5ad_path: Path, genes):
    """
    Compute expressing-cell fraction per region for a single *-log2.h5ad file.

    For each region r and gene g:
        expressing_fraction(r, g) =
            (# cells in region r with expression > 0 for gene g) /
            (total # cells in region r)

    Returns
    -------
    df : pandas.DataFrame
        Region × gene matrix of fractions (0–1). Index: region (normalized),
        columns: receptor genes.
    cellcounts : dict
        Mapping region → total # cells in that region (for this file only).
    """
    print(f"\n[Density] Processing {h5ad_path.name}")

    try:
        ad = sc.read_h5ad(h5ad_path, backed="r")
    except Exception as e:
        print(f"  Error loading file: {e}")
        return pd.DataFrame(), {}

    region_col = detect_region_column(ad.obs.columns)
    if region_col is None:
        print("  No region column found — skipped.")
        try:
            ad.file.close()
        except Exception:
            pass
        return pd.DataFrame(), {}

    regions_raw = ad.obs[region_col]
    regions = normalize_region_series(regions_raw)
    region_values = regions.unique().tolist()

    # Determine gene symbols present in this file
    if "gene_symbol" in ad.var.columns:
        symbols = list(ad.var["gene_symbol"].values)
    else:
        symbols = list(ad.var.index)

    present_genes = [g for g in genes if g in symbols]
    if len(present_genes) == 0:
        print("  None of the target genes found — skipped.")
        try:
            ad.file.close()
        except Exception:
            pass
        return pd.DataFrame(), {}

    gene_idx = {g: symbols.index(g) for g in present_genes}

    rows = []
    cellcounts = {}

    # Iterate over each region
    for rv in region_values:
        mask = (regions == rv).values
        idx = np.where(mask)[0]
        n_cells = len(idx)
        cellcounts[rv] = n_cells

        if n_cells == 0:
            continue

        frac_row = {}
        for g in present_genes:
            gi = gene_idx[g]
            col = ad.X[idx, gi]

            # Convert from sparse to dense if needed
            if hasattr(col, "toarray"):
                col = col.toarray().ravel()
            else:
                col = np.asarray(col).ravel()

            frac = np.count_nonzero(col > 0) / n_cells
            frac_row[g] = frac

        frac_row["_region"] = rv
        frac_row["_n_cells"] = n_cells
        rows.append(frac_row)

        # ---- Write calculation log (fraction-level) ----
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"Region: {rv}, n_cells: {n_cells}\n")
            for g in present_genes:
                f.write(f"  {g}: {frac_row[g]:.4f}\n")
            f.write("\n")

    try:
        ad.file.close()
    except Exception:
        pass

    if len(rows) == 0:
        return pd.DataFrame(), cellcounts

    df = pd.DataFrame(rows).set_index("_region")

    # Create a full gene-column matrix including missing ones as NaN
    out = pd.DataFrame(index=df.index, columns=genes, dtype=float)
    for g in present_genes:
        out[g] = df[g]
    out["_n_cells"] = df["_n_cells"]

    return out, cellcounts


# -----------------------------------------------------------------------------
# LOGISTIC REGRESSION (CB removed, Isocortex baseline)
# -----------------------------------------------------------------------------

def run_logistic_regression(files, genes, out_dir: Path):
    """
    Perform cell-level logistic regression for each receptor gene, excluding CB
    and using Isocortex as the baseline region.

    Model (per gene g):
        y_ij = 1 if expression_ij(g) > 0 else 0
        logit P(y_ij = 1) = β0 + Σ_k β_k * I(region_ij = R_k)

    where:
        - Regions with label "cb" (cerebellum) are excluded.
        - "isocortex" is the reference (baseline) region.
        - β_k represents the log-odds difference in expression relative to
          Isocortex for region R_k.

    The function pools cells across all files for each gene and returns a
    DataFrame of region-wise coefficients and also saves a CSV for downstream
    plotting.
    """
    print("\n[Logit] Collecting region labels (for CB removal and baseline)...")

    # First pass: collect all unique macro-region labels (normalized)
    all_regions = set()
    for fpath in files:
        ad = sc.read_h5ad(fpath, backed="r")
        region_col = detect_region_column(ad.obs.columns)
        if region_col is None:
            ad.file.close()
            continue
        regs = normalize_region_series(ad.obs[region_col])
        all_regions.update(regs.unique().tolist())
        ad.file.close()

    if "isocortex" not in all_regions:
        print("ERROR: 'isocortex' not found in region labels. Cannot set baseline.")
        return None

    # Define region levels for dummy coding (exclude 'cb')
    region_levels = sorted([r for r in all_regions if r != "cb"])
    # Ensure Isocortex is first (baseline)
    region_levels = ["isocortex"] + [r for r in region_levels if r != "isocortex"]

    print("  Region levels (CB removed):", region_levels)

    coef_records = []

    # Loop over genes (sequential; each iteration may be heavy)
    for gene in genes:
        print(f"\n[Logit] Gene: {gene}")

        y_all = []
        region_all = []

        # Pool cells across all files for this gene
        for fpath in files:
            try:
                ad = sc.read_h5ad(fpath, backed="r")
            except Exception as e:
                print(f"  Error loading {fpath.name}: {e}")
                continue

            region_col = detect_region_column(ad.obs.columns)
            if region_col is None:
                ad.file.close()
                continue

            regs = normalize_region_series(ad.obs[region_col])

            # Skip CB (cerebellum) cells
            mask_not_cb = regs != "cb"

            # Determine gene index if present
            if "gene_symbol" in ad.var.columns:
                symbols = list(ad.var["gene_symbol"].values)
            else:
                symbols = list(ad.var.index)

            if gene not in symbols:
                ad.file.close()
                continue

            gi = symbols.index(gene)
            col = ad.X[:, gi]

            if hasattr(col, "toarray"):
                col = col.toarray().ravel()
            else:
                col = np.asarray(col).ravel()

            # Binary expression (1 if >0 else 0)
            y = (col > 0).astype(int)

            # Apply CB mask
            y = y[mask_not_cb.values]
            regs_gene = regs[mask_not_cb]

            y_all.append(y)
            region_all.append(regs_gene.values)

            ad.file.close()

        if len(y_all) == 0:
            print("  No data across files — skipped.")
            continue

        y_all = np.concatenate(y_all)
        region_all = np.concatenate(region_all)

        # Safety: require both 0 and 1 to fit logistic regression
        if (y_all.sum() == 0) or (y_all.sum() == len(y_all)):
            print("  Expression is constant (all 0 or all 1) — skipped.")
            continue

        # Build design matrix with Isocortex as baseline
        df_gene = pd.DataFrame({"y": y_all, "region": region_all})
        # Keep only region levels we defined (should already exclude cb)
        df_gene = df_gene[df_gene["region"].isin(region_levels)]

        # Categorical with fixed categories to enforce baseline
        cat_region = pd.Categorical(df_gene["region"], categories=region_levels)
        X_region = pd.get_dummies(cat_region, drop_first=True)
        # Now columns correspond to all non-baseline regions in region_levels[1:].

        X = sm.add_constant(X_region.astype(float))
        y_vec = df_gene["y"].astype(float).values

        try:
            model = sm.Logit(y_vec, X)
            result = model.fit(disp=0)
        except Exception as e:
            print(f"  Logistic regression failed for {gene}: {e}")
            continue

        # Store coefficients for each non-baseline region
        for reg_name in region_levels[1:]:
            if reg_name in result.params.index:
                coef_records.append(
                    {
                        "gene": gene,
                        "region": reg_name,
                        "coef_logit_vs_isocortex": result.params[reg_name],
                    }
                )

        print("  Done.")

    if len(coef_records) == 0:
        print("No logistic regression coefficients were computed.")
        return None

    coef_df = pd.DataFrame(coef_records)
    coef_csv = out_dir / "logistic_region_coefficients_CB_removed_isocortex_baseline.csv"
    coef_df.to_csv(coef_csv, index=False)
    print(f"\n[Logit] Saved logistic coefficients → {coef_csv}")

    return coef_df


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    files = find_log2_files(H5AD_DIR)
    print(f"Found {len(files)} *-log2.h5ad files.")

    if len(files) == 0:
        sys.exit("No *-log2.h5ad files found. Please check H5AD_DIR.")

    # -------------------------------------------------------------------------
    # (A) Region-level expressing fractions → percentages → heatmap
    # -------------------------------------------------------------------------
    region_mats = []
    cellcount_total = {}

    for f in files:
        df, cc = compute_expressing_fraction(f, CANDIDATE_RECEPTOR_GENES)
        if not df.empty:
            region_mats.append(df)
            for k, v in cc.items():
                cellcount_total[k] = cellcount_total.get(k, 0) + v

    if len(region_mats) == 0:
        sys.exit("No region-level data produced in expressing-fraction step.")

    # Concatenate rows (regions may appear multiple times across files)
    all_df = pd.concat(region_mats, axis=0, sort=False).fillna(np.nan)
    genes_only = [g for g in CANDIDATE_RECEPTOR_GENES if g in all_df.columns]

    # Average expressing fraction per region (fractions are 0–1)
    frac_mean = all_df[genes_only].groupby(all_df.index).mean()

    # Sum cell counts per region
    all_df_ncells = all_df["_n_cells"].groupby(all_df.index).sum()
    frac_mean["_n_cells"] = all_df_ncells

    # Sort region names alphabetically
    frac_mean = frac_mean.sort_index()

    # Convert fractions to percentages and round to 3 decimals
    frac_pct = frac_mean[genes_only] * 100.0
    frac_pct = frac_pct.round(3)

    # Normalize per region so that each row sums to ~100%
    row_sums = frac_pct.sum(axis=1).replace(0, np.nan)
    comp_pct = frac_pct.div(row_sums, axis=0) * 100.0
    comp_pct = comp_pct.round(3)

    # Save region expressing fractions (raw percentages) and compositions
    csv_frac = OUT_DIR / "region_by_gene_expression_percentage.csv"
    frac_pct.to_csv(csv_frac)
    print(f"\nSaved region-level expressing percentages (unnormalized) → {csv_frac}")

    csv_comp = OUT_DIR / "region_receptor_subtype_ratio.csv"
    comp_pct.to_csv(csv_comp)
    print(f"Saved receptor-composition matrix (rows ≈ 100%) → {csv_comp}")

    # Save region-level cell counts
    cc_df = pd.DataFrame.from_dict(cellcount_total, orient="index", columns=["n_cells"])
    cc_df = cc_df.sort_values("n_cells", ascending=False)
    cc_csv = OUT_DIR / "region_cell_counts.csv"
    cc_df.to_csv(cc_csv)
    print(f"Saved region cell counts → {cc_csv}")

    # ---- Heatmap (Figure 1) ----
    hdf = comp_pct.dropna(how="all")
    if hdf.empty:
        sys.exit("No heatmap data available.")

    plt.figure(figsize=(11, max(4, 0.6 * hdf.shape[0])))
    sns.heatmap(
        hdf,
        cmap="viridis",
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Expressing-cell fraction per receptor subtype (%)"},
    )
    plt.title("Receptor composition per brain region (%)", fontsize=14)
    plt.ylabel("Brain region (macro-division)")
    plt.xlabel("Receptor subtype")

    fig1_path = OUT_DIR / "Figure1_receptor_heatmap.png"
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved heatmap → {fig1_path}")

    # -------------------------------------------------------------------------
    # (B) Logistic regression (CB removed, Isocortex baseline) → boxplot
    # -------------------------------------------------------------------------
    coef_df = run_logistic_regression(files, CANDIDATE_RECEPTOR_GENES, OUT_DIR)
    if coef_df is None:
        print("Logistic regression part skipped due to lack of coefficients.")
        print("\nAll processing completed (density + heatmap only).")
        return

    # ---- Boxplot (Figure 2) ----
    # We want one box per receptor gene, summarising the distribution of
    # log-odds coefficients across regions (vs Isocortex baseline).
    plt.figure(figsize=(14, 5))
    # Order genes as in the candidate list
    coef_df["gene"] = pd.Categorical(
        coef_df["gene"], categories=CANDIDATE_RECEPTOR_GENES, ordered=True
    )
    sns.boxplot(
        data=coef_df.sort_values("gene"),
        x="gene",
        y="coef_logit_vs_isocortex",
        showfliers=True,
    )
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Log-odds coefficient (relative to Isocortex)")
    plt.xlabel("Receptor subtype")
    plt.title("Coefficient distribution per receptor subtype (CB removed)")

    fig2_path = OUT_DIR / "Figure2_logit_coeff_boxplot_CB_removed.png"
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved logistic-coefficient boxplot → {fig2_path}")

    # -------------------------------------------------------------------------
    print("\nAll processing completed successfully.")
    print("Results stored in:", OUT_DIR)
    print("Density log saved →", LOG_FILE)


if __name__ == "__main__":
    main()
