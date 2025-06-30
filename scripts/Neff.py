import numpy as np
from scipy.spatial.distance import pdist,squareform
################
# note: if you are modifying the alphabet
# make sure last character is "-" (gap)
################
alphabet = "ARNDCQEGHILKMFPSTWYV-"
states = len(alphabet)
a2n = {}
for a,n in zip(alphabet,range(states)):
  a2n[a] = n
################

def aa2num(aa):
  '''convert aa into num'''
  if aa in a2n: return a2n[aa]
  else: return a2n['-']

def filt_gaps(msa,gap_cutoff=0.5):
  '''filters alignment to remove gappy positions'''
  tmp = (msa == states-1).astype(float) # gappy positions
  non_gaps = np.where(np.sum(tmp.T,-1).T/msa.shape[0] < gap_cutoff)[0]
  return msa[:,non_gaps],non_gaps

def get_eff(msa,eff_cutoff=0.8):
  '''compute effective weight for each sequence'''
  
  # pairwise identity
  msa_sm = 1.0 - squareform(pdist(msa,"hamming"))

  # weight for each sequence
  msa_w = (msa_sm >= eff_cutoff).astype(float)
  msa_w = 1/np.sum(msa_w,-1)

  return msa_w

def N_eff(seqs=None,
          seqs_int=None):
    '''converts list of sequences to msa'''
    if seqs is not None:
        msa_ori = []
        for seq in seqs:
            msa_ori.append([aa2num(aa) for aa in seq])
        msa_ori = np.array(msa_ori)
    elif seqs_int is not None:
        msa_ori = seqs_int 
    else:
        print("A msa array is needed.")
        exit()
    # remove positions with more than > 50% gaps
    msa, v_idx = filt_gaps(msa_ori,0.5)
    
    # compute effective weight for each sequence
    msa_weights = get_eff(msa,0.8)

    return np.sum(msa_weights)
     
