#!/usr/bin/bash

mkdir experiments/real_data/raw_data
cd experiments/real_data/raw_data

# === RAW
# gene annotations
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161824/suppl/GSE161824%5FA549%5FKRAS%2Erawcounts%2Egenes%2Ecsv%2Egz
gunzip GSE161824_A549_KRAS.rawcounts.genes.csv.gz 

# cell annotations
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161824/suppl/GSE161824%5FA549%5FKRAS%2Erawcounts%2Ecells%2Ecsv%2Egz
gunzip GSE161824_A549_KRAS.rawcounts.cells.csv.gz

# data matrix
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161824/suppl/GSE161824%5FA549%5FKRAS%2Erawcounts%2Ematrix%2Emtx%2Egz
gunzip GSE161824_A549_KRAS.rawcounts.matrix.mtx.gz

# === PROCESSED
# gene annotations
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161824/suppl/GSE161824%5FA549%5FKRAS%2Eprocessed%2Egenes%2Ecsv%2Egz
gunzip GSE161824_A549_KRAS.processed.genes.csv.gz

# cell annotations
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161824/suppl/GSE161824%5FA549%5FKRAS%2Eprocessed%2Ecells%2Ecsv%2Egz
gunzip GSE161824_A549_KRAS.processed.cells.csv.gz

wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161824/suppl/GSE161824%5FA549%5FKRAS%2Evariants2cell%2Ecsv%2Egz
gunzip GSE161824_A549_KRAS.variants2cell.csv.gz 

wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161824/suppl/GSE161824%5FA549%5FKRAS%2Eprocessed%2Ecells%2Emetadata%2Ecsv%2Egz
gunzip GSE161824_A549_KRAS.processed.cells.metadata.csv.gz

# data matrix
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161824/suppl/GSE161824%5FA549%5FKRAS%2Eprocessed%2Ematrix%2Emtx%2Egz
gunzip GSE161824_A549_KRAS.processed.matrix.mtx.gz


wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.survival.tsv
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.htseq_fpkm.tsv.gz
gunzip TCGA-LUAD.htseq_fpkm.tsv.gz