import pickle

RAW_DATA_FOLDER = "experiments/real_data/raw_data"
PROCESSED_DATA_FOLDER = "experiments/real_data/processed"
DISTRIBUTION_FOLDER = "experiments/real_data/semisynthetic_distributions"

ncontexts2variants = {
    10: f"{RAW_DATA_FOLDER}/mixed_variants_2_each.txt",
    19: f"{RAW_DATA_FOLDER}/mixed_variants_4_each.txt",
    50: f"{RAW_DATA_FOLDER}/significant50_variants.txt",
    83: f"{RAW_DATA_FOLDER}/all_variants.txt",
}


def get_solution_filename(num_obs, ncontexts):
    return f"{DISTRIBUTION_FOLDER}/sol_num_obs={num_obs},num_context={ncontexts}.pkl"


def get_solution(num_obs, ncontexts):
    filename = get_solution_filename(num_obs, ncontexts)
    return pickle.load(open(filename, "rb"))


def get_prolif_genes(num_obs):
    return pickle.load(open(f"{PROCESSED_DATA_FOLDER}/prolif_genes_num_obs={num_obs}.pkl", "rb"))