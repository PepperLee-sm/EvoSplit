import argparse
from evosplit.utils import load_fasta, write_fasta, lprint
import os
import numpy as np
from multiprocessing import Pool
import time

def multi_run(cmds,core_nums):
    print('Commands list length is ', len(cmds))
    
    # running cmds on different pools
    print('Running different commands on %s subprocesses...' % core_nums)
    p = Pool(core_nums)
    for cmd in cmds:
        p.apply_async(os.system, args=(cmd, ))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    
p = argparse.ArgumentParser()
p.add_argument("keyword", action="store", help="Keyword to call all generated MSAs.")
p.add_argument("-i", action='store', help='fasta/a3m file of original alignment.')
p.add_argument("-o", action="store", help='name of output directory to write MSAs to.')
p.add_argument("-MSA_cluster_dir", action='store', help='Directory of MSA cluster.')
p.add_argument("--gap_cutoff", type=float, default=0.25, help='Remove sequences with gaps representing more than this frac of seq.')

args = p.parse_args()

os.makedirs(args.o, exist_ok=True)
# 读取msa，按gap cutoff过滤
f = open(os.path.join(args.o, "%s.log"% args.keyword), 'w')
IDs, seqs = load_fasta(args.i, f, args.gap_cutoff)
if len(IDs) <= 1024:
    lprint(f"The depth of MSA filtered by gap cutoff is: {len(IDs)}. It's too low to extend MSA cluster.", f)
    exit()
lprint(f"The depth of MSA filtered by gap cutoff is: {len(IDs)}.", f)
data = dict(zip(IDs, seqs))
tmp_dir = os.path.join(args.o, "tmp")
seqs_qid = []
IDs_qid = []
for file in os.listdir(args.MSA_cluster_dir):
    if (not file.endswith(".a3m")) or (not file.startswith(args.keyword)):
        continue
    path = os.path.join(args.MSA_cluster_dir, file)
    IDs_cluster, seqs_cluster = load_fasta(path)
    for i, ID in enumerate(IDs_cluster):
        if ID not in IDs_qid:
            IDs_qid.append(ID)
            seqs_qid.append(seqs_cluster[i])
    
# # qid进行过滤
# L = len(seqs[0])
# N = len(seqs)
# seqs_qid, IDs_qid = QID_filter(seqs, IDs, num_seqs=1024, tmp_dir=tmp_dir)
# N = 1024

data_qid = dict(zip(IDs_qid, seqs_qid))

data_pool = {ID:seq for ID, seq in data.items() if ID not in data_qid.keys()}
IDs_pool = list(data_pool.keys())
seqs_pool = list(data_pool.values())
write_fasta(IDs_pool, seqs_pool, outfile=os.path.join(args.o, f"pool.a3m")) # 包含gap的MSA pool
write_fasta(IDs_pool, [s.replace('-','').upper() for s in seqs_pool], outfile=os.path.join(args.o, f"pool.a3m.fasta")) # 去掉gap & 大写

# 1024条序列在gapfilter msa中分别检索子msa
submsa_path = os.path.join(args.o, 'submsa')
if not os.path.exists(submsa_path):
    os.mkdir(submsa_path)
time0 = time.time()
jac_cmds = []
for index, ID in enumerate(IDs_qid):
    write_fasta([ID], [seqs_qid[index].replace('-','').upper()], outfile=submsa_path+'/'+str(index)+'.fasta')
    cmd = f'jackhmmer -A {submsa_path}/{index}.sto --noali --F1 0.0005 --F2 5e-05 --F3 5e-07 --incE 0.0001 -E 0.0001 --cpu 2 -N 1 {submsa_path}/{index}.fasta {args.o}/pool.a3m.fasta;~/tools/reformat.pl sto a3m {submsa_path}/{index}.sto {submsa_path}/{index}.a3m;rm -rf {submsa_path}/{index}.sto'
    jac_cmds.append(cmd)
multi_run(jac_cmds,8)
time1 = time.time()
lprint(f"Time of submsa searching: {time1-time0}", f)
# 遍历每个msa cluster，若深度>=64则跳过。否则进行扩充
lprint(f"Cluster\tMSA_Depth", f)
for file in os.listdir(args.MSA_cluster_dir):
    print(file)
    if (not file.endswith(".a3m")) or (not file.startswith(args.keyword)):
        continue
    path = os.path.join(args.MSA_cluster_dir, file)
    IDs_cluster, seqs_cluster = load_fasta(path)
    if len(IDs_cluster) >= 64:
        write_fasta(IDs_cluster, seqs_cluster, outfile=os.path.join(args.o, file))
        lprint(f"{file.split('.')[0]}\t{len(IDs_cluster)}", f)
        continue
    freq = {ID:0 for ID in IDs_pool}
    posi = {ID:0 for ID in IDs_pool}
    # 遍历cluster中的每条序列，读取submsa，记录出现次数和位置
    for ID_cluster in IDs_cluster[1:]: #排除query序列
        index = IDs_qid.index(ID_cluster)
        IDs_sub, seqs_sub = load_fasta(submsa_path+'/'+str(index)+'.a3m')
        for p, ID_sub in enumerate(IDs_sub[1:]): # 排除query序列
            if ID_sub in IDs_pool:
                freq[ID_sub] += 1
                posi[ID_sub] += p
            elif "/".join(ID_sub.split('/')[0:-1]) in IDs_pool:
                ID_sub = "/".join(ID_sub.split('/')[0:-1])
                freq[ID_sub] += 1
                posi[ID_sub] += p
            else:
                print("Error!", ID_sub)
    # 选择submsa的交集，若扩充后深度超过64，则取平均位置最前的序列
    IDs_common = [ID for ID in freq.keys() if freq[ID]==(len(IDs_cluster)-1)]
    if (len(IDs_common)+len(IDs_cluster)) <= 64:
        IDs_extend = IDs_cluster + IDs_common
        seqs_extend = seqs_cluster + [data_pool[ID] for ID in IDs_common]
        write_fasta(IDs_extend, seqs_extend, outfile=os.path.join(args.o, file))
    else:
        posi_common = {ID: posi[ID] for ID in IDs_common}
        topn = 64-len(IDs_cluster)
        IDs_common_topn = [pair[0] for pair in sorted(posi_common.items(), key = lambda kv:(kv[1], kv[0]))[0:topn]]
        IDs_extend = IDs_cluster + IDs_common_topn
        seqs_extend = seqs_cluster + [data_pool[ID] for ID in IDs_common_topn]
        write_fasta(IDs_extend, seqs_extend, outfile=os.path.join(args.o, file))
    lprint(f"{file.split('.')[0]}\t{len(IDs_extend)}", f)
                
        