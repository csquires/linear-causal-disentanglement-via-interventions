# === IMPORTS: BUILT-IN ===
import os
import pickle
from argparse import ArgumentParser

# === IMPORTS: THIRD-PARTY ===
import pandas as pd
from tqdm import tqdm
import numpy as np
from numpy.linalg import pinv

# === IMPORTS: LOCAL ===
from experiments.real_data.config import RAW_DATA_FOLDER, PROCESSED_DATA_FOLDER

# === ARGUMENT PARSING
parser = ArgumentParser()
parser.add_argument("--num_obs", type=int, default=100)
args = parser.parse_args()
NUM_OBS = args.num_obs

os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

# === LOAD GENE EXPRESSIONS AND METADATA
print("Loading gene expressions and metadata")
# 90,262 cells by 1,145 genes
ursu_gex_mat: pd.DataFrame = pd.read_pickle(f'{RAW_DATA_FOLDER}/ursu_gex_mat.pkl')

# === REMOVE UNASSIGNED CELLS
print("Removing unassigned cells")
# 150,044 * 103
v2c = pd.read_csv(f'{RAW_DATA_FOLDER}/GSE161824_A549_KRAS.variants2cell.csv', sep = '\t').set_index('cell')
v2c['synonymous'] = [(a[0] == a[-1]) for a in v2c['variant']]
# 115,068 * 104
var_cells = v2c[v2c['variant'] != 'unassigned']
var_cells['new_cat'] = ['syn' if (a[0]==a[-1]) else a for a in var_cells['variant']]
pickle.dump(var_cells, open(f"{PROCESSED_DATA_FOLDER}/var_cells.pkl", "wb"))


# === SELECT INDICES
print("Removing unassigned cells")
# 89,262 cells * 1,145 genes
common_inds = list(set(var_cells.index.values) & set(ursu_gex_mat.index.values))
x  = ursu_gex_mat.loc[common_inds]

# === SAVE SUBSET OF HIGH-VARIANCE GENES
# from enrichR
gene_set = 'CSF1;CD81;IRS2;CLU;MYC;NAMPT;CYP1B1;SOX9;JUNB;IER5;SOX4;IGFBP3;PGF;EREG;SULT2B1;AR;SFRP1;ADAM17;NUPR1;KIF20B;TP53;PDGFA;ZFP36L1;SDCBP;NUAK1;IGFBP7;IGFBP6;JUN;XRCC6;TFAP2C;JUND;JAG1;JUP;XRCC5;INSR;FN1;OSGIN1;IGF2;HMGA1;CDC6;FOSL1;BMP4;SQLE;RGCC;CDK6;BAMBI;FES;ID2;MDM2;MDM4;CALR;FGFR1;SLC35F6;BTG2;CDKN1A;BTG1;CXCL8;HILPDA;KIF14;HMGB2;LAMC2;CXCL1;AREG;CXCL5;RPS4X;HHEX;HNF4A;ITGAV;TIMP1;KLF10;IL11;EDN1;PRMT1;TESC;GKN1;NRG1;GREM1;SAPCD2;IRF1;TRNP1;BIRC5;MVD;SLC25A5;CDC123;PPM1D;TOB1;EGFR;PTHLH;SERTAD1;CAMK2N1;SH3BP4;HES1;S1PR3;TP53I11;NPM1;STAT1;AKR1C3;AKR1C2;SOD2;TBX3;FABP6;PRC1;DLC1;CTNNB1;KRAS;HSPA1A'.split(';')
prolif_genes = x.loc[:, gene_set].std(axis = 0).sort_values()[-NUM_OBS:].index.values
pickle.dump(prolif_genes, open(f"{PROCESSED_DATA_FOLDER}/prolif_genes_num_obs={NUM_OBS}.pkl", "wb"))

# === SAVE GENE EXPRESSION MATRIX
x = x.loc[~var_cells["new_cat"].isin({"multiple", "syn"})]
groups = x.groupby(var_cells['new_cat'].loc[common_inds])
pickle.dump(x, open(f"{PROCESSED_DATA_FOLDER}/expressions.pkl", "wb"))

# === COMPUTE COVARIANCE MATRICES
print("Computing covariance matrices")
# 84 cov_mats, each is NUM_OBS * NUM_OBS
cov_mats = {}
for variant_label, inds in tqdm(groups.groups.items()):   
    print(variant_label)
    data = x.loc[inds, prolif_genes].values
    mean_centered = data - data.mean(axis=0, keepdims=True)
    cov_mats[variant_label] = np.dot(mean_centered.T, mean_centered) / len(mean_centered)


# === COMPUTE PRECISION MATRICES
print("Computing precision matrices")
cov_obs = cov_mats['WT']
Theta_obs = pinv(cov_obs)

Theta_dict = {}
for variant_label, cov_mat in cov_mats.items():
    if variant_label != "WT":
        precision_mat = pinv(cov_mat)
        Theta_dict[variant_label] = precision_mat

# === SAVE DATA
data = dict(
    Theta_dict=Theta_dict, 
    Theta_obs=Theta_obs, 
    cov_mats=cov_mats,
)
pickle.dump(data, open(f"{PROCESSED_DATA_FOLDER}/thetas_num_obs={NUM_OBS}.pkl", "wb"))

