import os
import warnings
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import mudata as md
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage import color, filters, morphology, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopyWarning.*")

from src.loaders import load_universal_dataset
from src.qc_and_integration import preprocess_human_spatial, integrate_human_donors
from src.annotation import extract_cluster_markers, annotate_human_immune_cells
from src.signaling import infer_human_signaling
from src.ml_generalization import evaluate_spatial_classifier, evaluate_cross_donor_generalization

def clean_dataset_name(path):
    """Extracts the meaningful dataset name, skipping generic Cell Ranger / 10x subfolder names."""
    norm = os.path.normpath(path)
    if norm.lower().endswith(".gz"):
        norm = norm[:-3]

    # Walk upward if the path points to a generic 10x file or subfolder
    generic_names = {
        "matrix",
        "matrix.mtx",
        "filtered_feature_bc_matrix",
        "raw_feature_bc_matrix",
        "cell_feature_matrix",
        "spatial",
        "outs",
    }

    curr = norm
    base_no_ext, _ = os.path.splitext(os.path.basename(curr))
    while base_no_ext.lower() in generic_names:
        parent = os.path.dirname(curr)
        if not parent or os.path.basename(parent).lower() in {"", ".", "data"}:
            break
        curr = parent
        base_no_ext, _ = os.path.splitext(os.path.basename(curr))

    # Strip repetitive Cell Ranger suffixes for a clean folder name
    for suffix in ["_filtered_feature_bc_matrix", "_raw_feature_bc_matrix", "_sample_filtered_feature_bc_matrix"]:
        base_no_ext = base_no_ext.replace(suffix, "")

    return base_no_ext


def get_output_dirs(input_path=None, donor2_path=None, cohort_demo=False):
    """Creates dedicated results/<name>/ and figures/<name>/ folders for each run."""
    if cohort_demo:
        run_name = "demo_pbmc_cohort"
    elif input_path and donor2_path:
        run_name = f"{clean_dataset_name(input_path)}_vs_{clean_dataset_name(donor2_path)}"
    elif input_path:
        run_name = clean_dataset_name(input_path)
    else:
        run_name = "demo_10x_breast_cancer_visium"

    fig_dir = os.path.join("figures", run_name)
    res_dir = os.path.join("results", run_name)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    print(f"[Output] Saving figures to '{fig_dir}/' and tables to '{res_dir}/'")
    return fig_dir, res_dir


def run_histology_tiff_pipeline(img, fig_dir, res_dir):
    """Runs when load_universal_dataset returns a histology image array (.tif / .ome.tif)."""
    gray = color.rgb2gray(img) if img.ndim == 3 else img.astype(float)
    blurred = filters.gaussian(gray, sigma=1.5)
    mask = blurred < filters.threshold_otsu(blurred)
    mask = morphology.remove_small_holes(mask)
    mask = morphology.remove_small_objects(mask, max_size=15)

    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=4, labels=mask)
    peak_mask = np.zeros(distance.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)
    nuclei_labels = watershed(-distance, markers, mask=mask)

    props = measure.regionprops_table(
        nuclei_labels,
        intensity_image=gray,
        properties=["label", "area", "eccentricity", "mean_intensity"],
    )
    df_morph = pd.DataFrame(props)
    df_morph.to_csv(os.path.join(res_dir, "nuclei_morphology.csv"), index=False)
    print(f"[Histology] Segmented {len(df_morph)} nuclei — saved table to {res_dir}/nuclei_morphology.csv")

    # Map area and eccentricity back onto the image pixels for spatial morphology heatmaps
    max_lbl = int(nuclei_labels.max())
    area_lookup = np.zeros(max_lbl + 1, dtype=float)
    ecc_lookup = np.zeros(max_lbl + 1, dtype=float)
    area_lookup[df_morph["label"].to_numpy()] = np.clip(df_morph["area"].to_numpy(), 0, np.percentile(df_morph["area"], 95))
    ecc_lookup[df_morph["label"].to_numpy()] = df_morph["eccentricity"].to_numpy()

    area_map = np.ma.masked_where(nuclei_labels == 0, area_lookup[nuclei_labels])
    ecc_map = np.ma.masked_where(nuclei_labels == 0, ecc_lookup[nuclei_labels])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax1.set_title("Histology TIFF Image")
    ax1.axis("off")

    im2 = ax2.imshow(area_map, cmap="magma")
    ax2.set_title(f"Nuclear Area (n={len(df_morph)} Nuclei)")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    im3 = ax3.imshow(ecc_map, cmap="viridis")
    ax3.set_title("Nuclear Elongation (Eccentricity)")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(fig_dir, "histology_morphology_panels.png"), dpi=200, bbox_inches="tight")
    

def run_annotation_table_summary(df, fig_dir, res_dir):
    """Summarizes and plots cell-type annotation CSV tables (e.g., Xenium cell_groups.csv)."""
    text_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    label_col = next(iter(text_cols)) if text_cols else next(iter(df.columns))

    counts = df[label_col].astype(str).value_counts()
    summary_df = pd.DataFrame({
        "Cell_Group": counts.index,
        "Cell_Count": counts.values,
        "Percentage": ((counts.values / len(df)) * 100).round(2),
    })

    summary_df.to_csv(os.path.join(res_dir, "cell_group_counts.csv"), index=False)
    print(f"\n[Annotation Summary] Total annotated cells: {len(df)}")
    print(summary_df.to_string(index=False))

    plt.figure(figsize=(9, max(4, len(summary_df) * 0.45)))
    plt.barh(summary_df["Cell_Group"][::-1], summary_df["Cell_Count"][::-1], color="#2b5c8f")
    plt.xlabel("Number of Cells")
    plt.title("Cell Group Composition")
    plt.savefig(os.path.join(fig_dir, "cell_group_distribution.png"), dpi=200, bbox_inches="tight")
    

def run_multimodal_pipeline(mdata, fig_dir, res_dir, custom_genes=None, resolution=0.6):
    """Runs when load_universal_dataset returns a MuData object (e.g., TotalSeq-C / .h5mu)."""
    rna = mdata["rna"]
    prot = mdata["prot"]

    rna.var["mt"] = rna.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], inplace=True)
    keep = (rna.obs.n_genes_by_counts > 200) & (rna.obs.pct_counts_mt < 15)
    rna = rna[keep, :].copy()
    prot = prot[keep, :].copy()

    if rna.n_obs > 5000:
        idx = np.random.default_rng(42).choice(rna.n_obs, size=5000, replace=False)
        rna = rna[idx, :].copy()
        prot = prot[idx, :].copy()

    if rna.X.max() > 50:
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
    rna.raw = rna.copy()

    if prot.X.max() > 50:
        sc.pp.log1p(prot)

    rna = annotate_human_immune_cells(rna, model_name="Immune_All_High.pkl")
    print("\nAnnotated Multi-Modal Human Cell Types:")
    print(rna.obs["celltypist_label"].value_counts())

    markers_df = extract_cluster_markers(rna, groupby="celltypist_label", top_n=3)
    markers_df.to_csv(os.path.join(res_dir, "multimodal_markers.csv"), index=False)

    sc.pp.highly_variable_genes(rna, n_top_genes=2000)
    rna_hvg = rna[:, rna.var.highly_variable].copy()
    sc.pp.scale(rna_hvg, max_value=10)
    sc.tl.pca(rna_hvg)
    sc.pp.neighbors(rna_hvg)
    sc.tl.umap(rna_hvg)
    rna.obsm["X_pca"] = rna_hvg.obsm["X_pca"]

    mdata_out = md.MuData({"rna": rna, "prot": prot})
    h5mu_path = os.path.join(res_dir, "processed_multimodal.h5mu")
    mdata_out.write(h5mu_path)
    print(f"[MuData] Saved {h5mu_path}")

    # Multi-Modal ML Benchmark (RNA PCA + Surface Proteins)
    X_rna = rna.obsm["X_pca"]
    X_prot = prot.X.toarray() if hasattr(prot.X, "toarray") else np.asarray(prot.X)
    X_combined = np.hstack([X_rna, X_prot])
    y = rna.obs["celltypist_label"]

    X_tr, X_te, y_tr, y_te = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    print(f"[ML Benchmark] Multi-Modal (RNA + Protein) Accuracy: {accuracy_score(y_te, clf.predict(X_te)):.2%}")

    # Select custom genes OR top 2 RNA biomarker genes
    if custom_genes:
        top_genes = [g for g in custom_genes if g in rna.raw.var_names][:3]
    else:
        top_genes = []
        for g in markers_df["Marker_Gene"].tolist():
            if g in rna.raw.var_names and g not in top_genes:
                top_genes.append(g)
            if len(top_genes) == 2:
                break

    # Pick top 2 highest-expressed surface antibodies from Protein modality
    prot_means = np.asarray(prot.X.mean(axis=0)).ravel()
    top_prot_idx = np.argsort(prot_means)[::-1][:2]
    prot_cols = []
    for idx_p in top_prot_idx:
        p_name = str(prot.var_names[idx_p])
        col_label = f"Protein: {p_name}"
        vec = prot[:, idx_p].X.toarray().ravel() if hasattr(prot.X, "toarray") else np.asarray(prot[:, idx_p].X).ravel()
        rna_hvg.obs[col_label] = vec
        prot_cols.append(col_label)

    # Attach full .raw genes so use_raw=True works cleanly
    rna_hvg.raw = rna.raw.to_adata()
    rna_hvg.obs["celltypist_label"] = rna.obs["celltypist_label"]

    panels = ["celltypist_label"] + top_genes + prot_cols
    titles = ["CellTypist Immune Type"] + [f"RNA: {g}" for g in top_genes] + prot_cols

    sc.pl.umap(
        rna_hvg,
        color=panels,
        title=titles,
        ncols=3,
        wspace=0.45,
        use_raw=True,
        show=False,
    )
    plt.savefig(os.path.join(fig_dir, "multimodal_rna_protein_umap.png"), dpi=200, bbox_inches="tight")


def run_single_sample_pipeline(adata, fig_dir, res_dir, resolution=0.6):
    """Runs for any single-sample AnnData (.h5, .h5ad, .mtx, .parquet, or Visium spatial)."""
    is_spatial = "spatial" in adata.obsm
    # Grab table_type BEFORE subsample() can drop .uns keys
    table_type = adata.uns.get("table_type", "gene_expression")
    if adata.n_vars < 50 and table_type == "gene_expression":
        table_type = "visium_hd_bins"

    adata = preprocess_human_spatial(adata, resolution=resolution)

    if adata.n_obs > 5000:
        print(f"[CPU Speedup] Subsampling {adata.n_obs} -> 5,000 cells for fast analysis...")
        sc.pp.subsample(adata, n_obs=5000, random_state=42)

    output_config = {
        "gene_expression": ("cluster_markers.csv", "Marker_Gene", "Gene Expression Clusters"),
        "visium_hd_bins": ("visium_hd_bin_features.csv", "Subcellular_Bin_Feature", "Visium HD Cell Bin Clusters"),
        "cell_morphology": ("cell_morphology_features.csv", "Morphology_Feature", "Cell Morphology Clusters"),
        "tissue_positions": ("spatial_spot_grid_features.csv", "Spatial_Grid_Feature", "Tissue Spot Grid Clusters"),
    }
    csv_name, feature_col_name, _ = output_config.get(
        table_type, ("cluster_features.csv", "Spatial_Feature", "Spatial Clusters")
    )

    markers_df = extract_cluster_markers(adata, groupby="leiden", top_n=5, feature_col=feature_col_name)
    out_csv = os.path.join(res_dir, csv_name)
    markers_df.to_csv(out_csv, index=False)

    print(f"\nTop Features saved to {out_csv}:")
    print(markers_df.head(15).to_string(index=False))

    # 1. Pick top 3 unique biomarker genes discovered by Wilcoxon rank-sum test
    top_genes = []
    if table_type == "gene_expression":
        gene_col = feature_col_name
        for g in markers_df[gene_col].tolist():
            if g in adata.raw.var_names and g not in top_genes:
                top_genes.append(g)
            if len(top_genes) == 3:
                break

    # 2. Optional: Add CellTypist biological labels (e.g., Epithelial tumor cells vs Immune cells)
    if table_type == "gene_expression" and adata.n_vars >= 500:
        try:
            adata = annotate_human_immune_cells(adata, model_name="Immune_All_High.pkl")
            base_panels = ["leiden", "celltypist_label"]
        except Exception:
            base_panels = ["leiden"]
    else:
        base_panels = ["leiden", "nuclear_fraction"] if "nuclear_fraction" in adata.obs.columns else ["leiden"]

    panels_to_plot = base_panels + top_genes
    panel_titles = [p.replace("_", " ").title() if p in base_panels else f"Gene: {p}" for p in panels_to_plot]

    # 3. Plot multi-panel Spatial Scatter or UMAP showing Clusters + Cell Types + Specific Genes!
    if is_spatial:
        sq.pl.spatial_scatter(
            adata,
            color=panels_to_plot,
            title=panel_titles,
            ncols=3,
            use_raw=True,
            shape=None,
            size=2,
        )
        plt.savefig(os.path.join(fig_dir, f"{table_type}_spatial_genes.png"), dpi=200, bbox_inches="tight")
    else:
        sc.pl.umap(
            adata,
            color=panels_to_plot,
            title=panel_titles,
            ncols=3,
            wspace=0.45,
            use_raw=True,
            show=False,
        )
        plt.savefig(os.path.join(fig_dir, f"{table_type}_umap_genes.png"), dpi=200, bbox_inches="tight")
    

def run_donor_cohort_pipeline(fig_dir, res_dir, donor1_path=None, donor2_path=None, custom_genes=None, resolution=0.6):
    """Integrates any two donor datasets (.h5, .h5ad, .mtx) or built-in 10x human donors."""
    if donor1_path and donor2_path:
        d1 = load_universal_dataset(donor1_path)
        d2 = load_universal_dataset(donor2_path)
        if isinstance(d1, md.MuData):
            d1 = d1["rna"]
        if isinstance(d2, md.MuData):
            d2 = d2["rna"]
    else:
        d1, d2 = load_universal_dataset(input_path=None, mode="cohort")

    adata = integrate_human_donors(d1, d2, resolution=resolution)
    adata = annotate_human_immune_cells(adata, model_name="Immune_All_High.pkl")

    markers_df = extract_cluster_markers(adata, groupby="celltypist_label", top_n=3)
    markers_df.to_csv(os.path.join(res_dir, "cohort_donor_markers.csv"), index=False)
    print("\nAnnotated Human Cell Types Across Donors:")
    print(adata.obs["celltypist_label"].value_counts())

    evaluate_cross_donor_generalization(adata, donor_col="donor", label_col="celltypist_label")

    # Select custom genes OR top 2 shared cell-type marker genes
    if custom_genes:
        top_genes = [g for g in custom_genes if g in adata.raw.var_names][:2]
    else:
        top_genes = []
        for g in markers_df["Marker_Gene"].tolist():
            if g in adata.raw.var_names and g not in top_genes:
                top_genes.append(g)
            if len(top_genes) == 2:
                break

    # Panel 1: Before vs After ComBat & CellTypist
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    sc.pl.embedding(adata, basis="X_umap_uncorrected", color="donor", ax=ax1, show=False, title="Before Batch Correction (By Donor)")
    sc.pl.umap(adata, color="donor", ax=ax2, show=False, title="After ComBat (Donors Aligned)")
    sc.pl.umap(adata, color="celltypist_label", ax=ax3, show=False, title="CellTypist Human Annotation")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "multi_donor_cohort.png"), dpi=200, bbox_inches="tight")
    plt.show()

    # Panel 2: Specific Biomarker Genes Across Aligned Donors
    if top_genes:
        sc.pl.umap(
            adata,
            color=["celltypist_label"] + top_genes,
            title=["CellTypist Annotation"] + [f"Gene: {g}" for g in top_genes],
            ncols=3,
            wspace=0.45,
            use_raw=True,
            show=False,
        )
        plt.savefig(os.path.join(fig_dir, "multi_donor_genes_umap.png"), dpi=200, bbox_inches="tight")
        

def main():
    parser = argparse.ArgumentParser(description="Universal Human Tumor & Multi-Donor Atlas")
    parser.add_argument("--input", type=str, default=None, help="Path to any .h5, .h5ad, .h5mu, .csv, or 10x folder")
    parser.add_argument("--donor2", type=str, default=None, help="Optional second donor file (.h5, .h5ad, .mtx) for cohort integration")
    parser.add_argument("--cohort-demo", action="store_true", help="Run built-in 2-donor human PBMC cohort demo")
    parser.add_argument("--resolution", type=float, default=0.6, help="Leiden clustering resolution")
    parser.add_argument(
        "--celltypist-model",
        type=str,
        default="Immune_All_High.pkl",
        help="Pretrained CellTypist model (e.g., 'Immune_All_High.pkl', 'Cells_Adult_Breast.pkl')"
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        default=None,
        help="Optional list of specific genes to plot on UMAP/Spatial (e.g. --genes ERBB2 MKI67 CD8A)"
    )

    args = parser.parse_args()

    fig_dir, res_dir = get_output_dirs(input_path=args.input, donor2_path=args.donor2, cohort_demo=args.cohort_demo)

    # 1. Multi-Donor Cohort Mode
    if args.cohort_demo or (args.input and args.donor2):
        run_donor_cohort_pipeline(fig_dir, res_dir, donor1_path=args.input, donor2_path=args.donor2, resolution=args.resolution)
        return

    # 2. Single Universal Input
    data = load_universal_dataset(args.input, mode="spatial")
    if isinstance(data, pd.DataFrame):
        run_annotation_table_summary(data, fig_dir, res_dir)
    elif isinstance(data, np.ndarray):
        run_histology_tiff_pipeline(data, fig_dir, res_dir)
    elif isinstance(data, md.MuData):
        run_multimodal_pipeline(data, fig_dir, res_dir, resolution=args.resolution)
    else:
        run_single_sample_pipeline(data, fig_dir, res_dir, resolution=args.resolution)

if __name__ == "__main__":
    main()