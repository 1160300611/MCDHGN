# Preprocessing of Gene Expression

> In our study, we follow the data preprocessing steps as described in EMOGI[1], and the data preprocessing code is derived from [EMOGI](https://github.com/schulter/EMOGI). 
> 
> [1]Schulte-Sasse R, Budach S, Hnisz D, et al. Integration of multiomics data with graph convolutional networks to identify new 
cancer genes and their associated molecular mechanisms [J]. Nature Machine Intelligence, 2021, 3(6): 513-26.


## Overview

The `preprocess_gene_expression.py` script is designed for preprocessing gene expression data that has been quantified using the FPKM method (fragments per kilobase of exon model per million reads mapped). This script handles data from The Cancer Genome Atlas (TCGA) which includes both normal and tumor tissue samples, as well as normal tissue samples from the Genotype-Tissue Expression (GTEX) project. The data source for this script is published by Nature Scientific Data.

## Data Usage

In the processing pipeline:
- **Gene Expression Data**: Only the normal tissue data from GTEX and tumor tissue data from TCGA are utilized.
- **Calculation of Log2 Fold Changes**: The script computes log2 fold changes between these two datasets. This computation method is typically used to identify differentially expressed genes that may play significant roles in disease progression by comparing gene expression levels between normal and disease states.

## Data Download

The necessary gene expression datasets can be accessed and downloaded from the following Figshare link: [Data record 3](https://figshare.com/articles/dataset/Data_record_3/5330593). Users are required to download this data to run the script effectively.

## Core Functionality

### compute_geneexpression_foldchange Function

This function is at the core of the script and performs the following operations:
1. **Data Loading**: Reads the gene expression data for both tumor and normal tissues.
2. **Data Alignment**: Ensures that the gene indices (rows) of both datasets are aligned.
3. **Median Expression Calculation**: Computes the ratio of the median expression of genes in all tumor samples to that in all normal samples.
4. **Log2 Transformation**: Applies a log2 transformation to these ratios to derive log2 fold changes.

## Output Files

The script generates TSV files containing gene differential expression features for each cancer type analyzed. The file naming follows the pattern `GE_{CANCER_TYPE}_expression_matrix.tsv`, for example:
- `GE_BLCA_expression_matrix.tsv` for bladder cancer.
- `expression_mean_counts_16_gtexnormal.tsv` for multiple cancer types.

These files serve as a foundational resource for further biological analyses, particularly for studies investigating gene expression variations associated with cancer.

