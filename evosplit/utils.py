import numpy as np
from Bio import SeqIO
import os
import subprocess
from typing import List, Tuple
from scipy.spatial.distance import cdist


def lprint(string,f):
    print(string)
    f.write(string+'\n')

def load_fasta(fil, log_f=None, frac_gaps=1):
    seqs, IDs =[], []
    alignment = SeqIO.parse(fil, "fasta")
    query = next(alignment)
    query_seq = str(query.seq)
    query_length = len(query_seq)
    seqs.append(query_seq)
    IDs.append(query.id)
    gap_cutoff = query_length * frac_gaps
    remove = 0
    for record in alignment:
        seq = ''.join([x for x in record.seq if x.isupper() or x=='-']) # remove lowercase letters in alignment
        if (record.id not in IDs) and (seq.count('-') < gap_cutoff): # remove duplicate
            IDs.append(record.id)
            seqs.append(seq)
        else:
            remove += 1
    if log_f is not None:
        lprint("%d seqs removed for containing more than %d%% gaps, %d remaining" % (remove, int(frac_gaps*100), len(seqs)), log_f)
    return IDs, seqs

def write_fasta(names, seqs, outfile='tmp.fasta'):
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    with open(outfile,'w') as f:
        for nm, seq in list(zip(names, seqs)):
            f.write(">%s\n%s\n" % (nm, seq))
                        
# Select sequences from the MSA by QID filter
def QID_filter(seqs, IDs, num_seqs, tmp_dir, qid_min=0, qid_max=50):
    all_msa = f"{tmp_dir}/all_tmp.a3m"
    write_fasta(IDs, seqs, outfile=all_msa)
    for i in np.arange(qid_min,qid_max+1):
        filtered_msa = f"{tmp_dir}/msa_{i}.a3m"
        subprocess.check_output('hhfilter -i {:} -o {:}/msa_{:}.a3m -qid {:}'.format(all_msa, tmp_dir, i, i), shell=True)
        with open(filtered_msa, 'r') as file:
            msa_depth = sum(1 for line in file)/2
        if (msa_depth < num_seqs) and (i>0):
            chosen_qid = i-1
            break
        else:
            chosen_qid = i
    IDs_, seqs_ = load_fasta(f"{tmp_dir}/msa_{chosen_qid}.a3m")
    if len(IDs_) > num_seqs:
        seqs_, IDs_ = greedy_select(seqs_, IDs_, num_seqs=num_seqs, mode='max')
    os.system(f"rm -r {tmp_dir}")
    return seqs_, IDs_

def greedy_select(seqs: List, names:List, num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(seqs) <= num_seqs:
        return seqs, names
    
    array = np.array([list(seq) for seq in seqs], dtype=np.bytes_).view(np.uint8)
    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(seqs))
    indices = [0]
    pairwise_distances = np.zeros((0, len(seqs)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [seqs[idx] for idx in indices], [names[idx] for idx in indices]

