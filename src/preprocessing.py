
#import libs
import scanpy as sc
import anndata as ad
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", category = UserWarning)

# RNA preprocessing function
def preprocess_rna(filepath, cell_line_name, project_root = None, min_genes = 1000, max_counts = 100000, pct_mt_max = 20, n_hvgs = 2000, n_pcs = 50, n_neighbors = 15,
                   leiden_resolution = 0.5):
    """
    Full RNA preprocessing pipeline for a single neuroblastoma cell line.

    Parameters:
    filepath: str
        Path to the 10x .h5 file
    cell_line_name: str
        Name of the cell line (used for output filenames and obs labels)
    min_genes: int
        Minimum genes per cell (default 1000, based on Be2c optimization)
    max_counts: int
        Maximum total counts per cell (default 100000, for doublet removal)
    pct_mt_max: float
        Maximum mitochondrial percentage (default 20%)
    n_hvgs: int
        Number of highly variable genes to select (default 2000)
    n_pcs: int
        Number of PCs to compute (default 50)
    n_neighbors: int
        Neighbors for graph construction (default 15)
    leiden resolution: float
        Leiden clustering resolution (default 0.5)
    output_dir: str
        Where to save the processed .h5ad file

    Returns:
    adata: Anndata
        Fully processed AnnData object
    """
    # Resolve output directory
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok = True)
    
    print(f"\n{'='*50}")
    print(f"Processing: {cell_line_name}")
    print(f"{'='*50}")

    # Load
    adata = sc.read_10x_h5(filepath)
    adata.var_names_make_unique()
    adata.obs['cell_line'] = cell_line_name
    print(f"Loaded: {adata.shape}")

    # QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars = ['mt'], percent_top = None, log1p = False, inplace = True)

    # Try scrublet (run for reference but not hard filtering
    try:
        sc.pp.scrublet(adata)
        print(f"Scrublet predicted doublets: {adata.obs['predicted_doublet'].sum()}")
    except Exception as e:
        print(f"Scrublet skipped: {e}")

    # Filtering
    sc.pp.filter_cells(adata, min_genes = min_genes)
    sc.pp.filter_genes(adata, min_cells = 3)
    adata = adata[adata.obs.total_counts < max_counts].copy()
    adata = adata[adata.obs.pct_counts_mt < pct_mt_max].copy()
    print(f"After filtering: {adata.shape}")

    # Normalization
    sc.pp.normalize_total(adata, target_sum = 1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    # HVGs
    sc.pp.highly_variable_genes(adata, min_mean = 0.0125, max_mean = 3, min_disp = 0.5)
    print(f"HVGs: {adata.var.highly_variable.sum()}")
    adata = adata[:, adata.var.highly_variable].copy()
    
    # Scale + PCA
    # keep unscaled copy
    adata_unscaled = adata.copy()
    
    sc.pp.scale(adata, max_value = 10)
    sc.pp.pca(adata, n_comps = n_pcs)

    # Transfer PCA results to unscaled object
    adata_unscaled.obsm['X_pca'] = adata.obsm['X_pca']
    adata_unscaled.varm['PCs'] = adata.varm['PCs']
    adata_unscaled.uns['pca'] = adata.uns['pca']

    # Neighbors + UMAP + Clustering
    sc.pp.neighbors(adata, n_neighbors = n_neighbors, n_pcs = 15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution = leiden_resolution, flavor = "igraph", n_iterations = 2, directed = False)

    # Transfer neighbors/UMAP/clustering results to unscaled object
    adata_unscaled.obsm['X_umap'] = adata.obsm['X_umap']
    adata_unscaled.obs['leiden'] = adata.obs['leiden']
    adata_unscaled.uns['leiden'] = adata.uns['leiden']
    adata_unscaled.uns['umap'] = adata.uns['umap']
    adata_unscaled.obsp['distances'] = adata.obsp['distances']
    adata_unscaled.obsp['connectivities'] = adata.obsp['connectivities']
    adata_unscaled.uns['neighbors'] = adata.uns['neighbors']

    print(f"Clusters: {adata_unscaled.obs['leiden'].nunique()}")
    
    # Save unscaled object
    out_path = os.path.join(project_root, "data", "processed", f"{cell_line_name}_rna_processed.h5ad")
    print(f"Saving to: {out_path}")
    adata_unscaled.write_h5ad(out_path, compression = 'gzip')
    print(f"Saved successfully: {os.path.exists(out_path)}")
    
    return adata_unscaled
    