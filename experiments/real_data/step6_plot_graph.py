# === IMPORTS: BUILT-IN ===
import os
import pickle
from argparse import ArgumentParser

# === IMPORTS: THIRD-PARTY ===
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# === IMPORTS: LOCAL ===
from experiments.real_data.config import PROCESSED_DATA_FOLDER
from experiments.real_data.config import ncontexts2variants, get_prolif_genes, get_solution

# === ARGUMENT PARSING
parser = ArgumentParser()
parser.add_argument("--num_obs", type=int, default=83)
parser.add_argument("--num_contexts", type=int, default=83)
args = parser.parse_args()
NUM_OBS = args.num_obs
NUM_CONTEXTS = args.num_contexts

# === PLOTTING ===
FIGSIZE = (25, 6)
NODE_SIZE = 400
INTERVENTION_LABEL_SIZE = 6
LATENT_LABEL_SIZE = 6
INTERVENTION_X_OFFSET = 40
INTERVENTION_Y_OFFSET = 20

# === CUTOFFS ===
ALPHA = 0.1
EDGE_MIN = 0.2

# === READ SOLUTIONS
print("Reading solution")
variant_order = list(pd.read_csv(ncontexts2variants[NUM_CONTEXTS], header=None, index_col=0).index)
prolif_genes = get_prolif_genes(NUM_OBS)
sol = get_solution(NUM_OBS, NUM_CONTEXTS)
ix2target = sol["ix2target"]
target2ix = {target: ix for ix, target in ix2target.items()}

tcga_results = pickle.load(open(f"{PROCESSED_DATA_FOLDER}/tcga_results.pkl", "rb"))
tcga_pvals = tcga_results["adjusted_pvals"] 
tcga_coefs = tcga_results["coefs"]
significant_latents = np.where(tcga_pvals < ALPHA)[0]
significant_latent_labels = [variant_order[target2ix[l]] for l in significant_latents]

plt.clf()
plt.figure(figsize=FIGSIZE)
plt.axis("off")

# === CREATE GRAPH OVER LATENT NODES
A  = sol['B0_est']/np.diag(sol['B0_est'])[:,None]
nnodes = A.shape[0]
latent_labels = {i: f"$Z_{{{i+1}}}$" for i in range(nnodes)}
g = nx.DiGraph()
g.add_nodes_from(list(range(nnodes)))

# === COMPUTE POSITIONS PRIOR TO ADDING EDGES FROM INTERVENTION NODES
pruned = np.where(np.abs(A) > EDGE_MIN) #filter by abs value
weights = {}
strong_edges = []
for child, parent in zip(*pruned):
    if child == parent:
        continue #ignore the diagonal
    g.add_edge(parent, child)
    weights[parent, child] = A[child,parent]
    if A[child, parent] > EDGE_MIN:
        strong_edges.append((parent, child))
pos = nx.nx_agraph.graphviz_layout(g, prog="dot")


# === ADD INTERVENTION NODES
intervention_variants = variant_order[:nnodes]
intervention_labels = {variant: variant for variant in intervention_variants}
for ix, target in ix2target.items():
    g.add_edge(variant_order[ix], target)

    target_x, target_y = pos[target]
    pos[variant_order[ix]] = target_x - INTERVENTION_X_OFFSET, target_y + INTERVENTION_Y_OFFSET


# === DRAW SIGNIFICANT LATENT VARIABLES AND LABELS
nx.draw_networkx_nodes(
    g, 
    pos, 
    nodelist=significant_latents, 
    node_shape='o', 
    node_color="white",
    edgecolors="red",
    node_size=NODE_SIZE
)
nx.draw_networkx_labels(
    g, 
    pos, 
    latent_labels,
    font_size=LATENT_LABEL_SIZE
)
# === DRAW LATENT VARIABLES AND LABELS
nx.draw_networkx_nodes(
    g, 
    pos, 
    nodelist=list(set(range(nnodes)) - set(significant_latents)), 
    node_shape='o', 
    node_color="white",
    edgecolors="black",
    node_size=NODE_SIZE
)
nx.draw_networkx_labels(
    g, 
    pos, 
    latent_labels,
    font_size=LATENT_LABEL_SIZE
)
# === DRAW INTERVENTION VARIABLES AND LABELS
nx.draw_networkx_nodes(
    g, 
    pos, 
    nodelist=intervention_variants, 
    node_shape='s',
    node_color="lightgrey",
    edgecolors='black',
    node_size=NODE_SIZE
)
text = nx.draw_networkx_labels(
    g, 
    pos, 
    intervention_labels, 
    font_size=INTERVENTION_LABEL_SIZE,
    font_color="black",
    font_family="monospace"
)
# === DRAW EDGES
nx.draw_networkx_edges(
    g, 
    pos,
    # edgelist=strong_edges,
    arrowsize=10, 
    node_size=NODE_SIZE
)
plt.tight_layout()
os.makedirs("experiments/real_data/plots", exist_ok=True)
plt.savefig(f"experiments/real_data/plots/latent-graph-num_obs={NUM_OBS},num_contexts={NUM_CONTEXTS}.pdf")