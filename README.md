# EvoSplit

Code and data corresponding to "Disentangling Coevolutionary Constraints for Modeling Protein Conformational Heterogeneity". 

## Usage

### To generate MSA:

All MSAs used in this manuscript were generated using the [ColabFold notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb).

### To cluster MSA and generate subsampled MSA files:

`python scripts/run_evosplit.py EX -i initial_msa.a3m -o msas`

Outputs a directory named `msas` that contains

	- msas/EX_0.a3m
	- msas/EX_1.a3m
	...

`EX_0.a3m`, `EX_1.a3m` ... are the clusters identified by KMEANS.

### To run AF2:

`python scripts/RunAF2.py`

See https://github.com/jproney/AF2Rank for more information on compiling an AlphaFold2 installation.

### To run MSA Transformer:

`python scripts/runESM.py -i <my_subMSA.a3m> -o <outdir>`

### To calculate RMSD to provided reference structure(s):

`python scripts/CalculateModelFeatures.py path/to/pdbs/* -o <my_output_file>.json.zip --ref_struct REF_PDB_1.pdb REF_PDB_2.pdb`

