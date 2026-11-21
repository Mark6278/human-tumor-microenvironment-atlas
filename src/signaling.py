import liana as li

def infer_human_signaling(adata, groupby="leiden", top_n=15):
    """Infers human ligand-receptor cell-cell communication using LIANA."""
    print("[Signaling] Running LIANA human ligand-receptor analysis...")
    li.mt.rank_aggregate(
        adata,
        groupby=groupby,
        resource_name="consensus",
        use_raw=True,
        verbose=False
    )
    res = adata.uns["liana_res"].copy()
    cols = ["source", "target", "ligand_complex", "receptor_complex", "magnitude_rank", "specificity_rank"]
    available_cols = [c for c in cols if c in res.columns]
    return res[available_cols].head(top_n), res
