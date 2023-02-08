# === IMPORTS: BUILT-IN ===
import os

# === IMPORTS: THIRD-PARTY ===
import pandas as pd
from scipy.io import mmread
from experiments.real_data.config import RAW_DATA_FOLDER


if not os.path.exists(RAW_DATA_FOLDER):
    os.system("bash experiments/real_data/download.sh")

FORMAT_PROCESSED = True

if FORMAT_PROCESSED:
    print("Reading cells")
    cells = pd.read_csv(f'{RAW_DATA_FOLDER}/GSE161824_A549_KRAS.processed.cells.csv', header = None)[0].values
    print("Reading genes")
    genes = pd.read_csv(f'{RAW_DATA_FOLDER}/GSE161824_A549_KRAS.processed.genes.csv', header = None)[0].values
    print("Reading matrix")
    gexmat = mmread(f'{RAW_DATA_FOLDER}/GSE161824_A549_KRAS.processed.matrix.mtx')
    print("Converting to DataFrame")
    ursu_gex_mat = pd.DataFrame(gexmat.toarray(), index=cells, columns=genes)
    print("Pickling")
    ursu_gex_mat.to_pickle(f'{RAW_DATA_FOLDER}/ursu_gex_mat.pkl')

