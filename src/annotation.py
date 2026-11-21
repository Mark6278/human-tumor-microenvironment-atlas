import numpy as np
import pandas as pd
import scanpy as sc
import celltypist
from celltypist import models

def extract_cluster_markers(adata, groupby="leiden", top_n=5, feature_col="Marker_Gene"):
    """Extracts top marker genes or spatial features per cluster."""
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", use_raw=True)
    results = []

    for cluster_id in adata.obs[groupby].astype("category").cat.categories:
        markers_df = sc.get.rank_genes_groups_df(adata, group=cluster_id).head(top_n)
        genes = markers_df["names"].tolist()

        sub = adata.raw[adata.obs[groupby] == cluster_id, genes]
        total_obs = int(sub.n_obs)
        counts = np.asarray((sub.X > 0).sum(axis=0)).ravel()

        for gene, count, logfc, pval in zip(
            genes, counts, markers_df["logfoldchanges"], markers_df["pvals_adj"]
        ):
            results.append({
                "Cluster": cluster_id,
                "Total_Cells_or_Spots": total_obs,
                feature_col: gene,
                "Expressing_Count": int(count),
                "Expressing_Pct": round((count / total_obs) * 100, 1),
                "Log2FC": round(float(logfc), 2),
                "Adj_Pval": float(pval),
            })

    return pd.DataFrame(results)


def annotate_human_immune_cells(adata, model_name="Immune_All_High.pkl"):
    """Annotates human immune cells using pretrained CellTypist models on CPU."""
    print(f"[Annotation] Loading CellTypist human model: {model_name}...")
    models.download_models(model=model_name)
    model = models.Model.load(model=model_name)

    # Use clean log1p-normalized counts from .raw
    adata_for_ct = adata.raw.to_adata() if adata.raw is not None else adata.copy()

    # Convert sparse to dense float32 and clean any NaNs safely
    if hasattr(adata_for_ct.X, "toarray"):
        adata_for_ct.X = adata_for_ct.X.toarray()
    adata_for_ct.X = np.nan_to_num(adata_for_ct.X, nan=0.0, posinf=0.0, neginf=0.0)

    predictions = celltypist.annotate(adata_for_ct, model=model, majority_voting=True)
    adata.obs["celltypist_label"] = predictions.predicted_labels["majority_voting"].values
    return adata
