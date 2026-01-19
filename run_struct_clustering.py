import os
import json
import argparse
import pickle
import numpy as np
from sklearn import manifold
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from evosplit.align  import tmalign
from evosplit.utils  import load_fasta, write_fasta, lprint
from evosplit.structure import Structure
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

p = argparse.ArgumentParser()

p.add_argument("keyword", action="store", help="Keyword to call all generated MSAs.")
p.add_argument("--structure_dir", default="test/KaiB/AF2_results", help="Where to write output files")
p.add_argument("--output_dir", default="test/KaiB/struct_cluster", help="Where to write output files")
p.add_argument('--plddt_threshold', type=int, action='store',default=60, help='Step for epsilon scan for DBSCAN (Default 0.5).')
p.add_argument("--clustering_method", default="Hierarchical", help="Clustering method for AF2 structres. Hierarchical or DBscan")
p.add_argument('--distance_threshold', action='store', type=float, default=1.25, help="Distance_threshold if hierarchical method is chosed.")

args = p.parse_args()
os.makedirs(args.output_dir, exist_ok=True)
log_f = open(os.path.join(args.output_dir, "struct_clustering.log"), 'w')

np.random.seed(0)
num_struct = len([i for i in os.listdir(args.structure_dir) if i.endswith('.pdb')])
dirs = []
ids = []
plddts = []
for d in os.listdir(args.structure_dir):
    if (d.endswith('model_5_ptm_0_relaxed.pdb')) and (args.keyword in d):
        predict_path = f"{args.structure_dir}/{d}"
        struct = Structure(predict_path)
        plddt = struct.get_plddt()
        dirs.append(d)
        ids.append(d.split("/")[-1].split('_')[1])
        plddts.append(plddt)
tmscores = np.identity(len(dirs))
for i, id1 in enumerate(dirs):
    for j, id2 in enumerate(dirs[i+1:]):
        p1 = f"{args.structure_dir}/{id1}"
        p2 = f"{args.structure_dir}/{id2}"
        tmscores[i, i+j+1] = tmalign(p1, p2)
tmscores += tmscores.T - np.diag(tmscores.diagonal()) # symmetrization

with open(os.path.join(args.output_dir, f'structure_clustering_tmscore.pkl'), 'wb') as f:
    pickle.dump({"ids": ids, 'plddt': plddts, 'tmscore': tmscores}, f, protocol=4)
lprint(f"The lowest TMscore among models is {np.min(tmscores)}.", log_f)
if np.max(1/tmscores) <= args.distance_threshold:
    exit()

hc = AgglomerativeClustering(n_clusters=None, distance_threshold=args.distance_threshold, affinity='precomputed', linkage='average')
labels = hc.fit_predict(1/tmscores)
from scipy.cluster.hierarchy import linkage
distance_vector = 1/tmscores[np.triu_indices(len(tmscores), k=1)]

Z = linkage(distance_vector, method='average')

plt.figure(figsize=(10, 5))
dendrogram(Z,labels=ids)
plt.axhline(y=args.distance_threshold, color='r', linestyle='--')
plt.title('Hierarchical Clustering')
plt.xlabel('Sample id')
plt.ylabel('distance')
plt.savefig(f"{args.output_dir}/hierarchical_clustering_tmscore_{args.distance_threshold}.png")

clustering = {}
for label in set(labels):
    clustering[f"cluster_{label}"] = {}
    for i in np.where(labels==label)[0]:
        id = ids[i]
        clustering[f"cluster_{label}"][str(id)] = plddts[i]

with open(f"{args.output_dir}/structure_clustering_tmscores_{args.clustering_method}_{args.distance_threshold}.json", 'w') as f:
    json.dump(clustering, f, indent=2, sort_keys=True, ensure_ascii=False)
        
mds = manifold.MDS(n_components=2, dissimilarity='precomputed')
X_r=mds.fit_transform(1/tmscores)

plt.figure(figsize=(6,5))
colors = ['#f57c6e', '#71b7ed', '#f2b56f', '#f2a7da', '#fae69e', '#84c3b7', '#88d8db', '#b8aeeb', '#f2a7da', 'violet', 'slategrey', 'chocolate']
if np.where(labels==-1)[0].any():
    plt.scatter(X_r[np.where(labels==-1)[0], 0], X_r[np.where(labels==-1)[0], 1], c='gray', label='unclustered')
for i, label in enumerate(set(labels)):
    if label==-1:
        continue
    plt.scatter(X_r[np.where(labels==label)[0], 0], X_r[np.where(labels==label)[0], 1], c=colors[i], label=label)

plt.legend()
plt.xlabel("MDS1")
plt.ylabel("MDS2")

plt.savefig(f"{args.output_dir}/structure_clustering_{args.clustering_method}_{args.distance_threshold}.png")

f.close()