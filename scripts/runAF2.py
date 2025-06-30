import sys
import os
import argparse
import jax
import numpy as np
import pandas as pd
from scripts.utils import lprint
from scripts.align  import tmalign, get_rmsd
from scripts.structure import Structure
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument("input_msas", nargs='*', action='store', help="MSA to run prediction. First seq will be the predicted seq")
parser.add_argument("--recycles", type=int, default=1, help="Number of recycles when predicting")
parser.add_argument("--model_num", nargs='*', type=int, default=[1], help="Which AF2 model to use")
parser.add_argument("--seed", nargs='*', type=int, default=[0], help="RNG Seed")
parser.add_argument("--verbose", action='store_true', help="print extra")
parser.add_argument("--deterministic", action='store_true', help="make all data processing deterministic (no masking, etc.)")
parser.add_argument("--af2_dir", default="/lustre/grp/gyqlab/share/AF2_database/", help="AlphaFold code and weights directory")
parser.add_argument("--relax", action='store_true', help="Whether to relax")
parser.add_argument("--use_gpu_relax", action='store_true', help="Whether to relax on GPU. Relax on GPU can be much faster than CPU, so it is recommended to enable if possible. GPUs must be available if this setting is enabled.")
parser.add_argument("--output_dir", default="/home/haw053/metamorph_benchmark/", help="Where to write output files")
parser.add_argument("-gt1", default=None, help='pdb file of known ground truth conformation.')
parser.add_argument("-gt2", default=None, help='pdb file of known ground truth conformation.')
parser.add_argument("--run_PCA", action='store_true', help="Run PCA analysis on predicted structures")
parser.add_argument("--run_TSNE", action='store_true', help="Run TSNE analysis on predicted structures")

args = parser.parse_args()

sys.path.append(args.af2_dir)

os.makedirs(args.output_dir, exist_ok=True)
from alphafold.model import model
from alphafold.model import config
from alphafold.model import data

from alphafold.data import parsers
from alphafold.data import pipeline

from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.relax import relax

RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3
amber_relaxer = relax.AmberRelaxation(
  max_iterations=RELAX_MAX_ITERATIONS,
  tolerance=RELAX_ENERGY_TOLERANCE,
  stiffness=RELAX_STIFFNESS,
  exclude_residues=RELAX_EXCLUDE_RESIDUES,
  max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
  use_gpu=args.use_gpu_relax)
"""
Create an AlphaFold model runner
name -- The name of the model to get the parameters from. Options: model_[1-5]
"""
def make_model_runner(name, recycles):
  cfg = config.model_config(name)      

  cfg.data.common.num_recycle = recycles
  cfg.model.num_recycle = recycles
  cfg.data.eval.num_ensemble = 1
  if args.deterministic:
    cfg.data.eval.masked_msa_replace_fraction = 0.0
    cfg.model.global_config.deterministic = True
    
  params = data.get_model_haiku_params(name, args.af2_dir)

  return model.RunModel(cfg, params)

"""
Create a feature dictionary for input to AlphaFold
runner - The model runner being invoked. Returned from `make_model_runner`
sequence - The target sequence being predicted
templates - The template features being added to the inputs
seed - The random seed being used for data processing
"""
def make_processed_feature_dict(runner, a3m_file, name="test", templates=None, seed=0):
  feature_dict = {}

  # assuming sequence is first entry in msa

  with open(a3m_file,'r') as msa_fil:
    sequence = msa_fil.read().splitlines()[1].strip()

  feature_dict.update(pipeline.make_sequence_features(sequence, name, len(sequence)))

  with open(a3m_file,'r') as msa_fil:
    msa = pipeline.parsers.parse_a3m(msa_fil.read())

  feature_dict.update(pipeline.make_msa_features([msa]))

  if templates is not None:
    feature_dict.update(templates)
  else:
    feature_dict.update(empty_placeholder_template_features(num_templates=0, num_res=len(sequence)))


  processed_feature_dict = runner.process_features(feature_dict, random_seed=seed)

  return processed_feature_dict

"""
Make a set of empty features for no-template evalurations
"""
def empty_placeholder_template_features(num_templates, num_res):
  return {
      'template_aatype': np.zeros(
          (num_templates, num_res,
           len(residue_constants.restypes_with_x_and_gap)), dtype=np.float32),
      'template_all_atom_masks': np.zeros(
          (num_templates, num_res, residue_constants.atom_type_num),
          dtype=np.float32),
      'template_all_atom_positions': np.zeros(
          (num_templates, num_res, residue_constants.atom_type_num, 3),
          dtype=np.float32),
      'template_domain_names': np.zeros([num_templates], dtype=object),
      'template_sequence': np.zeros([num_templates], dtype=object),
      'template_sum_probs': np.zeros([num_templates], dtype=np.float32),
  }

"""
Package AlphaFold's output into an easy-to-use dictionary
prediction_result - output from running AlphaFold on an input dictionary
processed_feature_dict -- The dictionary passed to AlphaFold as input. Returned by `make_processed_feature_dict`.
"""
def parse_results(prediction_result, processed_feature_dict, relax=False):
  b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']  
  dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
  dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
  contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[:,:,dist_bins < 8].sum(-1)

  out = {"unrelaxed_protein": protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors),
        "plddt": prediction_result['plddt'],
        "pLDDT": prediction_result['plddt'].mean(),
        "dists": dist_mtx,
        "adj": contact_mtx}
  if relax:
    relaxed_pdb_str, _, violations = amber_relaxer.process(
        prot=out["unrelaxed_protein"])
    # relax_metrics[model_name] = {
    #     'remaining_violations': violations,
    #     'remaining_violations_count': sum(violations)
    # }
    out.update({'relaxed_protein': relaxed_pdb_str})
  out.update({"pae": prediction_result['predicted_aligned_error'],
              "pTMscore": prediction_result['ptm']})
  return out

def write_results(result, pdb_out_path, relax=False):
  plddt = float(result['pLDDT'])
  ptm = float(result["pTMscore"])
  print('plddt: %.3f' % plddt)
  print('ptm  : %.3f' % ptm)

  pdb_lines = protein.to_pdb(result["unrelaxed_protein"])
  with open(pdb_out_path, 'w') as f:
    f.write(pdb_lines)
  if relax:
    relaxed_pdb_lines = result["relaxed_protein"]
    relaxed_prefix = '_'.join(pdb_out_path.split("_")[0:-1])
    relaxed_pdb_out_path = f"{relaxed_prefix}_relaxed.pdb"
    with open(relaxed_pdb_out_path, 'w') as f:
      f.write(relaxed_pdb_lines)
  return plddt, ptm

model_runners = {}
for model_num in args.model_num:
  model_name = "model_{}_ptm".format(model_num)
  # results_key = model_name + "_seed_{}".format(args.seed)

  runner = make_model_runner(model_name, args.recycles)
  model_runners[model_name] = (runner)

log_f = open(f"{args.output_dir}/af2.log", 'w')
lprint(f'Have {len(model_runners)} models: {list(model_runners.keys())}', log_f)
lprint(f"name\tmodel\tseed\tplddt\tptm", log_f)
for input_msa in args.input_msas:
  name=os.path.basename(input_msa).replace('.a3m','')
  print(name)
  for model_name, runner in model_runners.items():
    for seed in args.seed:
      features = make_processed_feature_dict(runner, input_msa, name=name, seed=seed)
      result = parse_results(runner.predict(features, random_seed=seed), features, relax=args.relax)
      plddt, ptm = write_results(result, f"{args.output_dir}/{name}_{model_name}_{seed}_unrelaxed.pdb", relax=args.relax)
      lprint(f"{name}\t{model_name}\t{seed}\t{plddt}\t{ptm}", log_f)

if (args.gt1 is not None) and (args.gt2 is not None):
  # 计算TMscore、rmsd，可视化
  tmscore_gt1 = []
  tmscore_gt2 = []
  rmsd_gt1 = []
  rmsd_gt2 = []
  name = []
  plddts = []
  for d in sorted(os.listdir(args.output_dir)):
    if d.endswith('.pdb'):
      predict_path = os.path.join(args.output_dir, d)
      tmscore_gt1.append(tmalign(predict_path, args.gt1))
      tmscore_gt2.append(tmalign(predict_path, args.gt2))
      rmsd_gt1.append(get_rmsd(predict_path, args.gt1))
      rmsd_gt2.append(get_rmsd(predict_path, args.gt2))
      name.append(d.split('.')[0])
      struct = Structure(predict_path)
      plddts.append(struct.get_plddt())
    else:
      continue
  df_tmscore = pd.DataFrame({"tmscore_gt1": tmscore_gt1, "tmscore_gt2": tmscore_gt2, "plddt": plddts}, index=name)
  df_tmscore.to_csv(os.path.join(args.output_dir, "tmscore.csv"))
  df_rmsd = pd.DataFrame({"rmsd_gt1": rmsd_gt1, "rmsd_gt2": rmsd_gt2, "plddt": plddts}, index=name)
  df_rmsd.to_csv(os.path.join(args.output_dir, "rmsd.csv"))
  
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharex=True, sharey=True)
  im = ax.scatter(tmscore_gt1, tmscore_gt2, c=plddts, cmap="gist_rainbow")
  ax.set_xlabel("TMscore, Ground Truth 1")
  ax.set_ylabel("TMscore, Ground Truth 2")
  fig.colorbar(im, fraction=0.02, pad=0.05)
  plt.savefig(f"{args.output_dir}/AF2_tmscore.png")
  
  plt.figure(figsize=(6,5))
  plt.scatter(rmsd_gt1, rmsd_gt2, c=plddts, cmap="gist_rainbow")
  # plt.legend(bbox_to_anchor=(1,1), frameon=False)
  plt.xlabel("RMSD, Ground Truth 1")
  plt.ylabel("RMSD, Ground Truth 2")
  plt.colorbar(fraction=0.02, pad=0.05)
  plt.savefig(f"{args.output_dir}/AF2_rmsd.png")
  
# 按contact无监督聚类
if args.run_PCA or args.run_TSNE:
  contacts = []
  name = []
  plddts = []
  for d in sorted(os.listdir(args.output_dir)):
    if d.endswith('.pdb'):
      predict_path = os.path.join(args.output_dir, d)
      struct = Structure(predict_path)
      contacts.append(struct.pdb_to_contact().flatten())
      plddts.append(struct.get_plddt())
      name.append(d.split('.')[0])
    else:
      continue
  contacts = np.array(contacts)
if args.run_PCA:
  PCA_esm = PCA(2)
  contact_dr_PCA = PCA_esm.fit_transform(contacts)
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
  im = ax.scatter(contact_dr_PCA[:, 0], contact_dr_PCA[:, 1], c=plddts, cmap="gist_rainbow")
  ax.set_xlabel("PC 1")
  ax.set_ylabel("PC 2")
  fig.colorbar(im, fraction=0.02, pad=0.05)
  plt.savefig(f"{args.output_dir}/AF2_PCA.png")
if args.run_TSNE:
  contact_dr_TSNE = TSNE().fit_transform(contacts)
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
  im = ax.scatter(contact_dr_TSNE[:, 0], contact_dr_TSNE[:, 1], c=plddts, cmap="gist_rainbow")
  ax.set_xlabel("TSNE 1")
  ax.set_ylabel("TSNE 2")
  fig.colorbar(im, fraction=0.02, pad=0.05)
  plt.savefig(f"{args.output_dir}/AF2_TSNE.png")

log_f.close()