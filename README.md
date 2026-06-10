# Interpretable Neuroblastoma Multiome Analysis
BIG-TCR internship project

## Overview
This project explores the integration of paired single-nucleus RNA-seq (snRNA-seq) and ATAC-seq (snATAC-seq) data from neuroblastoma cell lines to identify regulatory mechanisms associated with MYCN amplification, a major driver of high-risk neuroblastoma.

Using the public multiome dataset GSE262189 (~42,000 nuclei across six neuroblastoma cell lines), the project builds and evaluates deep learning models that learn a joint RNA–ATAC latent representation of cellular states. The primary goal is to determine whether chromatin accessibility information provides additional biological and predictive value beyond gene expression alone.
### Key Research Question
Does incorporating chromatin accessibility (ATAC) data improve the identification and interpretation of MYCN-associated neuroblastoma states compared to RNA data alone?

## Project Structure
```
├── data/                  # Raw and processed AnnData/muon objects
├── notebooks/             # Exploratory analysis and visualization
├── preprocessing/         # RNA and ATAC preprocessing pipelines
├── models/                # MultiVI baseline + custom VAE
├── evaluation/            # Benchmarking scripts
├── interpretability/      # Peak-to-gene and TF motif analyses
└── README.md
```

## Objectives
Preprocess and analyze paired snRNA-seq and snATAC-seq data.
Build multimodal data structures using AnnData/MuData.
Learn joint latent representations of RNA and ATAC modalities.
Establish a multimodal baseline using MultiVI.
Compare RNA-only, ATAC-only, and RNA+ATAC models.
Evaluate model performance using MYCN amplification status.
Interpret learned biological signals through:
Peak-to-gene linking
Transcription factor motif accessibility
MYCN-associated regulatory programs

## Dataset
Dataset: GSE262189
Source: Gene Expression Omnibus (GEO)
Samples: Six neuroblastoma cell lines
Modalities: Paired snRNA-seq and snATAC-seq
Size: ~42,000 nuclei
Labels: MYCN-amplified vs. non-amplified cell lines

## Methods
### 1. RNA Preprocessing
- Filtering and QC
- Normalization and log-transformation
- Highly variable gene selection
- PCA, neighborhood graph, clustering (Scanpy)
### 2. ATAC Preprocessing
- Peak filtering
- TF-IDF normalization
- Latent semantic indexing (LSI)
- Chromatin accessibility analysis
- Integration into a paired muon object
### 3. Deep Learning Models
- MultiVI Baseline
- Multimodal variational inference (scvi-tools)
- Joint RNA+ATAC latent embedding
- Visualization by MYCN status
### 4. Model Comparison
We compare:
- RNA-only embedding
- ATAC-only embedding
- RNA+ATAC multimodal embedding
Evaluation metrics:
- Separation of MYCN-amplified vs non-amplified lines
- Classification accuracy (if supervised)
- Latent space clustering structure

## Tools/Skills learned
- Python
- Scanpy
- AnnData
- Muon
- scvi-tools / MultiVI
- PyTorch
- Jupyter Notebooks

## Expected Deliverables
- Reproducible preprocessing pipeline
- Multimodal integration workflow
- MultiVI baseline model
- Comparative evaluation of RNA-only, ATAC-only, and RNA+ATAC approaches
- Biological interpretation of MYCN-associated regulatory mechanisms
- Final report and presentation summarizing findings

## Learning Outcomes
This project combines single-cell genomics, multimodal data integration, and deep learning to develop practical experience in computational biology, interpretable machine learning, and regulatory genomics.
