import scanpy as sc
import anndata as ad

def preprocess_human_spatial(adata, resolution=0.6):
    """Runs human QC, normalization, PCA, and Leiden clustering."""
    # 1. Handle low-feature spatial metadata tables (e.g., Visium HD barcode_mappings.parquet)
    if adata.n_vars < 50:
        if adata.n_obs > 5000:
            print(f"[CPU Speedup] Subsampling {adata.n_obs} -> 5,000 cells for fast CPU clustering...")
            sc.pp.subsample(adata, n_obs=5000, random_state=42)

        adata.raw = adata.copy()
        sc.pp.scale(adata)

        n_comps = min(5, adata.n_obs - 1, adata.n_vars - 1)
        n_neighbors = min(15, adata.n_obs - 1)

        sc.tl.pca(adata, n_comps=n_comps)
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=n_neighbors)
        # init_pos="random" prevents UMAP spectral memory spikes on low-feature integer tables
        sc.tl.umap(adata, init_pos="random")
        sc.tl.leiden(adata, resolution=min(resolution, 0.3), flavor="igraph", n_iterations=2, directed=False)
        return adata

    # 2. Standard full-transcriptome gene expression QC
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs.pct_counts_mt < 20, :].copy()

    if adata.X.max() > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    adata.raw = adata.copy()

    n_top = min(2000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg, max_value=10)

    n_comps = min(30, adata_hvg.n_obs - 1, adata_hvg.n_vars - 1)
    n_neighbors = min(15, adata_hvg.n_obs - 1)

    sc.tl.pca(adata_hvg, n_comps=n_comps)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)
    return adata



def integrate_human_donors(donor1, donor2, resolution=0.6):
    """Concatenates two human donors, runs QC if raw counts, and removes batch effects."""
    adata = ad.concat(
        {"Donor_A": donor1, "Donor_B": donor2},
        label="donor",
        join="inner"
    )

    # Check if input is raw integer counts from 10x .h5 files
    if adata.X.max() > 50:
        print("[QC] Raw 10x counts detected — filtering low-quality & high-MT cells...")
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata = adata[adata.obs.pct_counts_mt < 15, :].copy()
    else:
        import numpy as np
        if hasattr(adata.X, "toarray"):
            adata.X = adata.X.toarray()
        adata.X = np.expm1(adata.X)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    # Select top 2,000 highly variable genes for fast CPU PCA & ComBat
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata_hvg = adata[:, adata.var.highly_variable].copy()

    # Uncorrected embedding (before batch correction)
    sc.tl.pca(adata_hvg)
    sc.pp.neighbors(adata_hvg, use_rep="X_pca")
    sc.tl.umap(adata_hvg)
    adata.obsm["X_umap_uncorrected"] = adata_hvg.obsm["X_umap"].copy()

    # ComBat batch correction across donors
    print("[Integration] Running ComBat batch correction across 10x human donors...")
    sc.pp.combat(adata_hvg, key="donor")
    sc.tl.pca(adata_hvg)
    adata.obsm["X_pca_harmony"] = adata_hvg.obsm["X_pca"].copy()

    sc.pp.neighbors(adata, use_rep="X_pca_harmony")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)
    return adata
