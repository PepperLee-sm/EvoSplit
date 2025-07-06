import subprocess
import shutil
import os
import numpy as np
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.Emboss.Applications import NeedleCommandline

from pymol import cmd
backbone_atoms = "N+CA+C+O"
def get_rmsd(pdb1, pdb2, cycles=0):
    cmd.load(pdb1, 'obj1')
    cmd.load(pdb2, 'obj2')
    align = cmd.align('obj1','obj2', cycles=cycles)
    cmd.delete('all')
    return align[0]
  
def get_backbone_rmsd(pdb1, pdb2, cycles=0):
    cmd.load(pdb1, 'obj1')
    cmd.load(pdb2, 'obj2')
    try:
      align = cmd.align(f'obj1 and name {backbone_atoms} and polymer',
                        f'obj2 and name {backbone_atoms} and polymer', cycles=cycles)
      cmd.delete('all')
      return align[0]
    except:
      cmd.delete('all')
      return 0
  
def tmalign(pdb1, pdb2):
    p1 = subprocess.Popen('TMalign ' + pdb1 + ' ' + pdb2 + " | grep 'if normalized by length of Chain_1' | awk '{print $2}'", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    p1.wait()
    if p1.poll() != 0:
        print("TMalign Failed! ")
    p2 = subprocess.Popen('TMalign ' + pdb1 + ' ' + pdb2 + " | grep 'if normalized by length of Chain_2' | awk '{print $2}'", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    p2.wait()
    if p2.poll() != 0:
        print("TMalign Failed! ")
    # p = subprocess.Popen('TMalign ' + pdb1 + ' ' + pdb2 + " | grep RMSD | awk '{print $5}'", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # p.wait()
    # if p.poll() != 0:
    #     print("TMalign Failed! ")
    # rmsd = p.stdout.read().decode('utf-8').replace(",", "")
    # return float(rmsd)
    tmscore1 = p1.stdout.read().decode('utf-8').replace("\n", "")
    tmscore2 = p2.stdout.read().decode('utf-8').replace("\n", "")
    return max(float(tmscore1), float(tmscore2))

def mmalign(pdb1, pdb2):
    p1 = subprocess.Popen('MMalign ' + pdb1 + ' ' + pdb2 + " | grep 'if normalized by length of Chain_1' | awk '{print $2}'", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    p1.wait()
    if p1.poll() != 0:
        print("MMalign Failed! ")
    p2 = subprocess.Popen('MMalign ' + pdb1 + ' ' + pdb2 + " | grep 'if normalized by length of Chain_2' | awk '{print $2}'", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    p2.wait()
    if p2.poll() != 0:
        print("MMalign Failed! ")
    # p = subprocess.Popen('TMalign ' + pdb1 + ' ' + pdb2 + " | grep RMSD | awk '{print $5}'", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # p.wait()
    # if p.poll() != 0:
    #     print("TMalign Failed! ")
    # rmsd = p.stdout.read().decode('utf-8').replace(",", "")
    # return float(rmsd)
    tmscore1 = p1.stdout.read().decode('utf-8').replace("\n", "")
    tmscore2 = p2.stdout.read().decode('utf-8').replace("\n", "")
    return max(float(tmscore1), float(tmscore2))
    
def align_seqs(seq1, seq2, needle_path=shutil.which('needle'), tmp_dir="./tmp"):
    """Align 2 sequences.

    Args:
        Query seq, seq2

    Returns:
        Aligned seq (str)
    """
    if not needle_path:
        raise ValueError("Needle path does not exist!")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    rec1 = SeqRecord(Seq(seq1), id="1")
    rec2 = SeqRecord(Seq(seq2), id="2")
    faa1 = f"{tmp_dir}/seq1.faa"
    faa2 = f"{tmp_dir}/seq2.faa"
    SeqIO.write(rec1, faa1, "fasta")
    SeqIO.write(rec2, faa2, "fasta")
    needle_cline1 = NeedleCommandline(needle_path,
                                 asequence=faa1,
                                 bsequence=faa2,
                                 gapopen=10, gapextend=0.5, outfile=f"{tmp_dir}/needle.txt")
    stdout1, stderr1 = needle_cline1()
    align1 = AlignIO.read(f"{tmp_dir}/needle.txt", "emboss")
    os.system(f"rm -r {tmp_dir}")
    return str(align1[0].seq), str(align1[1].seq)

def get_CA_rmsd(positions1, positions2, L=None, include_L=True, weights=None, copies=1):
#   true = positions1[:, 1, :]
#   pred = positions2[:, 1, :]
  return _get_rmsd_loss(positions1, positions2, weights=weights, L=L, include_L=include_L, copies=copies)

def _get_rmsd_loss(true, pred, weights=None, L=None, include_L=True, copies=1):
  '''
  get rmsd + alignment function
  align based on the first L positions, computed weighted rmsd using all 
  positions (if include_L=True) or remaining positions (if include_L=False).
  '''
  # normalize weights
  length = true.shape[-2]
  if weights is None:
    weights = (np.ones(length)/length)[...,None]
  else:
    weights = (weights/(weights.sum(-1,keepdims=True) + 1e-8))[...,None]

  # determine alignment [L]ength and remaining [l]ength
  if copies > 1:
    if L is None:
      L = iL = length // copies; C = copies-1
    else:
      (iL,C) = ((length-L) // copies, copies)
  else:
    (L,iL,C) = (length,0,0) if L is None else (L,length-L,1)

  # slice inputs
  if iL == 0:
    (T,P,W) = (true,pred,weights)
  else:
    (T,P,W) = (x[...,:L,:] for x in (true,pred,weights))
    (iT,iP,iW) = (x[...,L:,:] for x in (true,pred,weights))

  # get alignment and rmsd functions
  (T_mu,P_mu) = ((x*W).sum(-2,keepdims=True)/W.sum((-1,-2)) for x in (T,P))
  aln = _np_kabsch((P-P_mu)*W, T-T_mu)   
  align_fn = lambda x: (x - P_mu) @ aln + T_mu
  msd_fn = lambda t,p,w: (w*np.square(align_fn(p)-t)).sum((-1,-2))
  
  # compute rmsd
  if iL == 0:
    msd = msd_fn(true,pred,weights)
  elif C > 1:
    # all vs all alignment of remaining, get min RMSD
    iT = iT.reshape(-1,C,1,iL,3).swapaxes(0,-3)
    iP = iP.reshape(-1,1,C,iL,3).swapaxes(0,-3)
    imsd = msd_fn(iT, iP, iW.reshape(-1,C,1,iL,1).swapaxes(0,-3))
    imsd = (imsd.min(0).sum(0) + imsd.min(1).sum(0)) / 2 
    imsd = imsd.reshape(np.broadcast_shapes(true.shape[:-2],pred.shape[:-2]))
    msd = (imsd + msd_fn(T,P,W)) if include_L else (imsd/iW.sum((-1,-2)))
  else:
    msd = msd_fn(true,pred,weights) if include_L else (msd_fn(iT,iP,iW)/iW.sum((-1,-2)))
  rmsd = np.sqrt(msd + 1e-8)

  #return {"rmsd":rmsd, "align":align_fn}
  return rmsd

def _np_kabsch(a, b, return_v=False):
  '''get alignment matrix for two sets of coodinates'''
  ab = a.swapaxes(-1,-2) @ b
  u, s, vh = np.linalg.svd(ab, full_matrices=False)
  flip = np.linalg.det(u @ vh) < 0
  u_ = np.where(flip, -u[...,-1].T, u[...,-1].T).T
  u[...,-1] = u_
  return u if return_v else (u @ vh)