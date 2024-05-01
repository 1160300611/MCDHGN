# Preprocessing Mutation Frequencies
> In our study, we follow the data preprocessing steps as described in EMOGI[1], and the data preprocessing code is derived from [EMOGI](https://github.com/schulter/EMOGI). 
> 
> [1]Schulte-Sasse R, Budach S, Hnisz D, et al. Integration of multiomics data with graph convolutional networks to identify new 
cancer genes and their associated molecular mechanisms [J]. Nature Machine Intelligence, 2021, 3(6): 513-26.

## Preprocessing of SNVs

## Overview

The `preprocess_mutation_freqs.py` script is designed to preprocess single nucleotide variants (SNVs) from Mutation Annotation Format (MAF) files. This script is capable of normalizing SNV frequencies based on the exonic gene length provided that GENCODE annotation is available. It outputs a mean mutation frequency matrix across various cancer types (gene x cancer type matrix).

## Prerequisites

Before running the `preprocess_mutation_freqs.py` script, users must download and prepare the necessary data files:

### Mutation Annotation Format (MAF) Files

- Download relevant MAF files from The Cancer Genome Atlas (TCGA).
- The names of all MAF files utilized in the study are listed in the `gdc_manifest.mutations.mutect2.2018-11-23.txt` file.

### GENCODE Annotation

- Extract the `gencode.v28.basic.annotation.gff3` file from the `gencode.v28.basic.annotation.gff3.gz` GZIP file for gene length normalization purposes.

## Execution

Ensure that all required files, including MAF files and GENCODE annotations, are correctly downloaded and extracted as per the guidelines provided in the `gdc_manifest.mutations.mutect2.2018-11-23.txt`. Follow the example provided in the script to generate the mutation frequency matrix for a cancer type.

## Output

The script returns a matrix detailing mean mutation frequencies per cancer type, formatted as a gene x cancer type matrix and saved in a TSV format.

