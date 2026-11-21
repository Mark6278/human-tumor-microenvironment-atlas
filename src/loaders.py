import os
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import mudata as md
import anndata as ad
import tifffile
from scipy.sparse import csr_matrix

def _load_parquet_to_anndata(parquet_path):
    """Builds a properly tagged AnnData from any 10x Xenium or Visium HD .parquet file."""
    print(f"[Loader] Reading Parquet spatial table: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Case 1: Subcellular mRNA molecule table (transcripts.parquet) -> Real Gene Expression
    if {"feature_name", "x_location", "y_location", "cell_id"}.issubset(df.columns):
        if "qv" in df.columns:
            df = df[df["qv"] >= 20]
        df = df[df["cell_id"].astype(str) != "UNASSIGNED"].copy()

        cells_cat = df["cell_id"].astype(str).astype("category")
        genes_cat = df["feature_name"].astype(str).astype("category")

        counts = csr_matrix(
            (np.ones(len(df), dtype=np.float32), (cells_cat.cat.codes, genes_cat.cat.codes)),
            shape=(len(cells_cat.cat.categories), len(genes_cat.cat.categories)),
        )
        centroids = df.groupby(cells_cat, observed=True)[["x_location", "y_location"]].mean()

        adata = ad.AnnData(
            X=counts,
            obs=pd.DataFrame(index=cells_cat.cat.categories),
            var=pd.DataFrame(index=genes_cat.cat.categories),
        )
        adata.obsm["spatial"] = centroids.loc[adata.obs_names].to_numpy()
        adata.uns["table_type"] = "gene_expression"
        print(f"[Loader] Built spatial gene matrix from transcripts.parquet: {adata.n_obs} cells x {adata.n_vars} genes")
        return adata

    # Case 2: Visium HD Cell Segmentation Barcode Mapping (barcode_mappings.parquet) -> Subcellular Bin Metrics
    if {"square_002um", "cell_id", "in_nucleus", "in_cell"}.issubset(df.columns):
        print("[Loader] Detected Visium HD 'barcode_mappings.parquet' — aggregating subcellular squares per cell...")
        valid_df = df[df["in_cell"]].copy()
        cell_summary = valid_df.groupby("cell_id", observed=True).agg(
            total_002um_squares=("square_002um", "count"),
            nuclear_squares=("in_nucleus", "sum"),
            unique_008um_bins=("square_008um", "nunique"),
            unique_016um_bins=("square_016um", "nunique"),
        )
        cell_summary["cytoplasmic_squares"] = cell_summary["total_002um_squares"] - cell_summary["nuclear_squares"]
        cell_summary["nuclear_fraction"] = (cell_summary["nuclear_squares"] / cell_summary["total_002um_squares"]).round(3)

        cell_summary.index = cell_summary.index.astype(str)
        num_cols = cell_summary.columns.tolist()
        adata = ad.AnnData(
            X=cell_summary.to_numpy(dtype=np.float32),
            obs=cell_summary.copy(),
            var=pd.DataFrame(index=num_cols),
        )
        adata.uns["table_type"] = "visium_hd_bins"
        print(f"[Loader] Aggregated {adata.n_obs} segmented Visium HD cells across {len(num_cols)} spatial bin features")
        return adata

    # Case 3: Cell Morphology (cells.parquet) or Spot Positions (tissue_positions.parquet)
    if "in_tissue" in df.columns:
        df = df[df["in_tissue"] == 1].copy()

    id_col = next((c for c in ["cell_id", "barcode"] if c in df.columns), None)
    x_col = next((c for c in ["x_centroid", "pxl_col_in_fullres", "x"] if c in df.columns), None)
    y_col = next((c for c in ["y_centroid", "pxl_row_in_fullres", "y"] if c in df.columns), None)

    if id_col and x_col and y_col:
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        # Exclude raw pixel coordinates from clustering features so clusters reflect cell/nucleus morphology
        exclude_cols = {x_col, y_col, "in_tissue", "array_row", "array_col"}
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        # Fallback if tissue_positions.parquet only has grid/pixel coordinates
        if not feat_cols:
            feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "in_tissue"]

        adata = ad.AnnData(
            X=df[feat_cols].fillna(0).to_numpy(dtype=np.float32),
            obs=df.copy(),
            var=pd.DataFrame(index=feat_cols),
        )
        adata.obsm["spatial"] = df[[x_col, y_col]].to_numpy()
        adata.uns["table_type"] = "cell_morphology" if "cell_area" in df.columns else "tissue_positions"
        print(f"[Loader] Built '{adata.uns['table_type']}' AnnData: {adata.n_obs} entries x {len(feat_cols)} features")
        return adata

    raise ValueError(f"Unrecognized Parquet schema in {parquet_path}. Columns found: {df.columns.tolist()}")


def _load_tiff_image(tif_path):
    """Loads a pyramid level (< 2500 px) from a .tif / .ome.tif microscopy image safely on CPU."""
    print(f"[Loader] Reading histology TIFF image: {tif_path}")
    with tifffile.TiffFile(tif_path) as tif:
        series = next(iter(tif.series))
        img = series.asarray()
        for level in series.levels:
            if max(level.shape) <= 2500:
                img = level.asarray()
                break

    # Move RGB channel axis (smallest axis of size 1, 3, or 4) to the last axis (H, W, C)
    if img.ndim == 3:
        min_axis = int(np.argmin(img.shape))
        if img.shape[min_axis] <= 4 and min_axis != 2:
            img = np.moveaxis(img, min_axis, -1)

    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)

    print(f"[Loader] Loaded image slice with shape: {img.shape}")
    return img


def load_universal_dataset(input_path=None, mode="spatial"):
    """
    Universal auto-detecting loader for single-cell, spatial, multi-modal, Parquet, TIFF, and compressed files.
    """
    if input_path is None:
        if mode == "cohort":
            print("[Loader] No local path provided — loading built-in 10x human PBMC cohort...")
            d1 = sc.datasets.pbmc3k_processed().raw.to_adata()
            d2 = sc.datasets.pbmc68k_reduced().raw.to_adata()
            d1.var_names_make_unique()
            d2.var_names_make_unique()
            return d1, d2
        else:
            print("[Loader] No local path provided — fetching 10x Human Breast Cancer Visium dataset...")
            adata = sc.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
            adata.var_names_make_unique()
            return adata

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # 1. DIRECTORY input (10x Visium spatial folder OR 10x .mtx folder)
    if os.path.isdir(input_path):
        if os.path.exists(os.path.join(input_path, "spatial")):
            print(f"[Loader] Detected 10x Visium spatial directory: {input_path}")
            adata = sq.read.visium(input_path)
        else:
            print(f"[Loader] Detected 10x .mtx directory: {input_path}")
            adata = sc.read_10x_mtx(input_path, var_names="gene_symbols", cache=True)
        adata.var_names_make_unique()
        return adata

    # 2. FILE input (handle archives and strip .gz if compressed)
    lower_path = input_path.lower()
    if lower_path.endswith(".tar.gz") or lower_path.endswith(".tgz") or lower_path.endswith(".zip"):
        raise ValueError(
            f"'{os.path.basename(input_path)}' is a multi-file archive. Extract it into a folder first and pass the folder path."
        )

    clean_path = lower_path[:-3] if lower_path.endswith(".gz") else lower_path
    _, ext = os.path.splitext(clean_path)

    if ext == ".h5":
        print(f"[Loader] Detected 10x HDF5 (.h5) file: {input_path}")
        adata_all = sc.read_10x_h5(input_path, gex_only=False)
        adata_all.var_names_make_unique()

        sibling_cells = os.path.join(os.path.dirname(input_path), "cells.parquet")
        if os.path.exists(sibling_cells) and "spatial" not in adata_all.obsm:
            print("[Loader] Found sibling Xenium 'cells.parquet' — attaching (x, y) spatial coordinates!")
            cells_df = pd.read_parquet(sibling_cells)
            cells_df["cell_id"] = cells_df["cell_id"].astype(str)
            cells_df = cells_df.set_index("cell_id").reindex(adata_all.obs_names)
            adata_all.obsm["spatial"] = cells_df[["x_centroid", "y_centroid"]].to_numpy()

        if "feature_types" in adata_all.var.columns:
            modalities = adata_all.var["feature_types"].unique().tolist()
            if len(modalities) > 1:
                print(f"[Loader] Multi-modal .h5 detected ({modalities}) — building MuData container...")
                rna = adata_all[:, adata_all.var["feature_types"] == "Gene Expression"].copy()
                other_mods = [m for m in modalities if m != "Gene Expression"]
                prot = adata_all[:, adata_all.var["feature_types"].isin(other_mods)].copy()
                return md.MuData({"rna": rna, "prot": prot})

        return adata_all

    elif ext == ".h5ad":
        print(f"[Loader] Detected AnnData (.h5ad) file: {input_path}")
        adata = sc.read_h5ad(input_path)
        if "feature_name" in adata.var.columns and adata.var_names[0].startswith("ENSG"):
            adata.var_names = adata.var["feature_name"].astype(str)
        adata.var_names_make_unique()
        return adata

    elif ext == ".h5mu":
        print(f"[Loader] Detected MuData (.h5mu) file: {input_path}")
        return md.read(input_path)

    elif ext == ".parquet":
        return _load_parquet_to_anndata(input_path)

    elif ext in [".tif", ".tiff"] or clean_path.endswith(".ome.tif") or clean_path.endswith(".ome.tiff"):
        return _load_tiff_image(input_path)

    elif ext == ".mtx":
        from scipy.io import mmread

        print(f"[Loader] Detected Matrix Market (.mtx) file: {input_path}")
        mat = mmread(input_path).T.tocsr()
        adata = ad.AnnData(X=mat)
        adata.var_names_make_unique()
        return adata

    elif ext in [".csv", ".tsv", ".txt"]:
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df_raw = pd.read_csv(input_path, sep=sep, header=None)

        if len(df_raw) < 15:
            raise ValueError(
                f"'{os.path.basename(input_path)}' only has {len(df_raw)} rows (shape {df_raw.shape}). "
                "It appears to be a transformation matrix or summary table rather than a cell/sample dataset."
            )

        df = pd.read_csv(input_path, sep=sep, index_col=0)
        num_df = df.select_dtypes(include=[np.number])

        if len(num_df.columns) < 3:
            print(f"[Loader] Detected cell-group annotation table ({ext}): {input_path}")
            return df

        print(f"[Loader] Detected numeric count/feature table ({ext}): {input_path}")
        adata = ad.AnnData(
            X=num_df.fillna(0).to_numpy(dtype=np.float32),
            obs=df.copy(),
            var=pd.DataFrame(index=num_df.columns.astype(str)),
        )
        if adata.n_obs > 10000 and adata.n_vars < adata.n_obs:
            adata = adata.T
        adata.var_names_make_unique()
        return adata

    else:
        raise ValueError(
            f"Unsupported format '{ext}' for '{os.path.basename(input_path)}'. "
            "Supported: .h5, .h5ad, .h5mu, .parquet, .tif/.tiff, .mtx, .csv/.tsv (.gz), or 10x folders."
        )
