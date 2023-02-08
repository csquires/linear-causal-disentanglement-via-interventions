# === IMPORTS: BUILT-IN ===
import pickle
from argparse import ArgumentParser

# === IMPORTS: THIRD-PARTY ===
import pandas as pd
import numpy as np
from tqdm import trange
from statsmodels.stats.multitest import fdrcorrection
from lifelines import CoxPHFitter

# === IMPORTS: LOCAL ===
from experiments.real_data.config import RAW_DATA_FOLDER, PROCESSED_DATA_FOLDER
from experiments.real_data.config import get_solution, get_prolif_genes

# === ARGUMENT PARSING
parser = ArgumentParser()
parser.add_argument("--num_obs", type=int, default=50)
parser.add_argument("--num_contexts", type=int, default=50)
args = parser.parse_args()
NUM_OBS = args.num_obs
NUM_CONTEXTS = args.num_contexts

# === READ SOLUTIONS
print("Reading solution")
prolif_genes = get_prolif_genes(NUM_OBS)
sol = get_solution(NUM_OBS, NUM_CONTEXTS)

# === READ TCGA DATA
print("Reading TCGA data")
rna = pd.read_csv(f'{RAW_DATA_FOLDER}/TCGA-LUAD.htseq_fpkm.tsv', sep='\t')
surv = pd.read_csv(f'{RAW_DATA_FOLDER}/TCGA-LUAD.survival.tsv', sep='\t')
pats = set(rna.columns[1:]).intersection(surv['sample'].values)
surv = surv.set_index('sample').loc[pats]

# === CONVERT TO ENSEMBL IDs
rna['gene'] = [a.split('.')[0] for a in rna['Ensembl_ID']]
names = pd.read_csv(f'{RAW_DATA_FOLDER}/prolif_ensembl').set_index('initial_alias')
prolif_ensemb = names['converted_alias'].loc[prolif_genes].values

# === COMPUTE LATENT EXPRESSIONS
print("Computing latent expressions")
gex = rna.loc[:,pats].groupby(rna['gene']).sum().T
gex = pd.DataFrame(
    gex.values - gex.values.mean(axis=0, keepdims=True), 
    columns=gex.columns, 
    index=gex.index.values
)
prolif_gex = gex.loc[:, prolif_ensemb]
M = sol["H_est"] @ prolif_gex.values.T
latent_nodes = pd.DataFrame(M.T, index=gex.index.values)

# === COMPUTE P-VALUES
print("Computing p-values")
pvals = []
coefs = []
for i in trange(latent_nodes.shape[1]):
    dat = pd.concat([
        latent_nodes.loc[:,i], 
        surv['OS'].astype(int),
        surv['OS.time'].astype(float)
    ], axis =1)
    cph = CoxPHFitter()
    cph.fit(dat, 'OS.time', event_col='OS')
    pvals.append(cph.summary['p'].iloc[0])
    coefs.append(cph.summary['coef'].iloc[0])

adjusted_pvals = fdrcorrection(np.array(pvals))[1]
sig05 = np.sum(adjusted_pvals < 0.05)
sig10 = np.sum(adjusted_pvals < 0.1)
print(f"Number with p value < 0.05: {sig05}")
print(f"Number with p value < 0.1: {sig10}")
results = dict(
    coefs=coefs,
    adjusted_pvals=adjusted_pvals
)
pickle.dump(results, open(f"{PROCESSED_DATA_FOLDER}/tcga_results.pkl", "wb"))


# === COMPUTE P-VALUES FOR BASELINE
print("Computing p-values")
pvals_original = []
coefs_original = []
for i in trange(prolif_gex.shape[1]):
    dat = pd.concat([
        prolif_gex.iloc[:, i], 
        surv['OS'].astype(int),
        surv['OS.time'].astype(float)
    ], axis =1)
    cph = CoxPHFitter()
    cph.fit(dat, 'OS.time', event_col='OS')
    pvals_original.append(cph.summary['p'].iloc[0])
    coefs_original.append(cph.summary['coef'].iloc[0])

adjusted_pvals_original = fdrcorrection(np.array(pvals_original))[1]
sig05 = np.sum(adjusted_pvals_original < 0.05)
sig10 = np.sum(adjusted_pvals_original < 0.1)
print(f"Number with p value < 0.05: {sig05}")
print(f"Number with p value < 0.1: {sig10}")