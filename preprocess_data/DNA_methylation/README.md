# Preprocessing DNA Methylation Data

> In our study, we follow the data preprocessing steps as described in EMOGI[1], and the data preprocessing code is derived from [EMOGI](https://github.com/schulter/EMOGI). 
> 
> [1]Schulte-Sasse R, Budach S, Hnisz D, et al. Integration of multiomics data with graph convolutional networks to identify new 
cancer genes and their associated molecular mechanisms [J]. Nature Machine Intelligence, 2021, 3(6): 513-26.

# DNA Methylation Data Preprocessing Guide

This guide provides a comprehensive workflow for preprocessing DNA methylation data derived from the 450k Illumina bead arrays, using a series of scripts and tools designed to facilitate the analysis of such data from sources like TCGA and GTEX.

## Data Acquisition

Start by downloading the necessary DNA methylation data using the `gdc-client` tool:

```bash
gdc-client download -m gdc_manifest.2018-11-16.txt
```

This command will download the methylation data for normal and tumor samples as specified in the provided GDC files.

## Compute Average Methylation
Use the get_mean_sample_meth.py script to calculate the methylation matrices for both tumor and normal samples. This script enables the definition of gene promoters and computes methylation by mapping each CpG site to its corresponding gene based on the distance to the transcription start site (TSS).
```bash
python get_mean_sample_meth.py --annotation <path-to-annotation-gff3 file> --methylation-dir <path-to-downloaded-TCGA-methylation-data> --output <path-to-gene-sample-matrix>
```
## Batch Correction
For correcting batch effects, utilize the provided batchcorrection.ipynb notebook and an accompanying R script. These tools implement batch correction using ComBat, which normalizes methylation data against batch variables such as plate numbers. Ensure all R dependencies are installed prior to running the script.

## Computing Differential DNA Methylation
To compute differential DNA methylation values across promoters and gene bodies, use the preprocess_dna_methy.py script. This script subtracts normal from tumor beta values to calculate differential methylation.

## Output Files
The scripts generate files containing differential DNA methylation features, saved in TSV format. These feature matrices are crucial for subsequent analyses aiming to identify methylation changes associated with disease states.

## Additional Information
The methylation expression data files are large, and users should ensure adequate storage capacity before downloading. The manifests provided (gdc_manifest.dna_meth.solidnormal.2018-11-16.txt and gdc_manifest.dna_meth.tumor.2018-11-16.txt) contain the names of all files utilized in the study.

For further details on each script and additional options, refer to the specific documentation within each script file.