import argparse
import torch
from evosplit.utils import QID_filter, load_fasta, write_fasta, lprint
from evosplit.structure import Structure
from evosplit.infer import msatr, apc, symmetrize, map_top, map_filter, match_score, weight_filter
from evosplit.align import align_seqs
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA
import math
from sklearn.cluster import DBSCAN, KMeans
from sklearn import manifold


if __name__=='__main__':

    p = argparse.ArgumentParser(description=
    """
    Clustering sequences by disentangling coevolutionary signals belonging to different conformations.
    Shimian Li
    """)

    p.add_argument("keyword", action="store", help="Keyword to call all generated MSAs.")
    p.add_argument("-i", action='store', help='fasta/a3m file of original alignment.')
    p.add_argument("-o", action="store", help='name of output directory to write MSAs to.')
    p.add_argument("--gap_cutoff", action='store', type=float, default=0.25, help='Remove sequences with gaps representing more than this frac of seq.')
    p.add_argument("--subfamily_MSA_depth", action='store', type=int, default=1024, help='Depth of subfamily MSA.')
    p.add_argument("-gt1", action='store', default=None, help='pdb file of known ground truth conformation.')
    p.add_argument("-gt2", action='store', default=None, help='pdb file of known ground truth conformation.')
    p.add_argument("--topL", action='store', type=float, default=15/2, help='Number of coevolved amino acid pairs (@L) retained to be analysed. If =7.5, 7.5L pairs.')
    p.add_argument("--mean_cluster", action='store', type=int, default=32, help='Mean number of sequences of clusters')
    p.add_argument("--cluster_method", action="store", default="kmeans", help='Method of clustering.')
    p.add_argument('--eps_val', action='store', type=float, help="Use single value for eps instead of scanning.")
    p.add_argument('--min_eps', action='store',default=3, help='Min epsilon value to scan for DBSCAN (Default 3).')
    p.add_argument('--max_eps', action='store',default=20, help='Max epsilon value to scan for DBSCAN (Default 20).')
    p.add_argument('--eps_step', action='store',default=.5, help='step for epsilon scan for DBSCAN (Default 0.5).')
    
    
    args = p.parse_args()

    
    os.makedirs(args.o, exist_ok=True)
    f = open(os.path.join(args.o, "%s.log"% args.keyword), 'w')
    IDs, seqs = load_fasta(args.i, f, args.gap_cutoff)
    
    query = seqs[0]
    
    tmp_dir = os.path.join(args.o, "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    # if the depth of msa exceeds 1024, filter the sequences with qid
    L = len(seqs[0])
    N = len(seqs)
    if N > args.subfamily_MSA_depth:
        seqs, IDs = QID_filter(seqs, IDs, num_seqs=args.subfamily_MSA_depth, tmp_dir=tmp_dir)
        N = len(seqs)
    lprint(f"The length of query sequence is {L}.\nThe depth of MSA to be analysed is {N}.", f)
    # write all msa to fasta file
    write_fasta(IDs, seqs, outfile=os.path.join(args.o, f"{args.keyword}.a3m"))

    data = list(zip(IDs, seqs))
    alphabet, results = msatr(data)
    contacts = results['contacts'].cpu()[0]
    contacts_ = contacts.clone()
    contacts_filtered, filter_id = map_top(contacts_, args.topL)
    top_contacts = contacts_filtered.numpy()
    top_contacts[np.where(top_contacts>0)] = 1
    
    row_att = results['row_attentions'].cpu()[:, -1, :, :, :]
    row_att_all = results['row_attentions_all'].cpu()
    if alphabet.prepend_bos:
        row_att_all = row_att_all[:, 0, :, 1:, 1:]
        row_att = row_att[..., 1:, 1:]
        
    # symmetrization and apc
    row_att_apc = apc(symmetrize(row_att))
    row_att_all_apc = apc(symmetrize(row_att_all))

    # set the diagonal elements to 0
    diag_m = torch.ones((L, L))-torch.eye(L)
    row_att_apc_diag = torch.mul(row_att_apc, diag_m)
    row_att_all_apc_diag = torch.mul(row_att_all_apc, diag_m)
    
    row_att_apc_sum = torch.einsum('abcd->cd', row_att_apc_diag)
    row_att_apc_sum_ = row_att_apc_sum.clone()
    row_att_apc_sum_filtered, filter_id = map_top(row_att_apc_sum_, args.topL)
    row_att_apc_sum_top = row_att_apc_sum_filtered.numpy()
    row_att_apc_sum_top[np.where(row_att_apc_sum_top>0)] = 1
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 4), sharex=True, sharey=True)
    ax = ax.flatten()
    ax[0].imshow(contacts, origin='lower', cmap='Blues')
    ax[1].imshow(top_contacts, origin='lower', cmap='Blues')
    ax[0].set_title("prob")
    ax[1].set_title("contacts (top 15/2L)")
    plt.savefig(f"{args.o}/msatr_contact.png")
    
    s, h = row_att_all_apc_diag.shape[0:2]
    row_att_all_apc_norm_topn = torch.zeros_like(row_att_all_apc_diag)
    for i in range(s):
        for j in range(h):
            row_att_all_apc_norm_topn[i, j] = map_filter(row_att_all_apc_diag[i, j], filter_id)
    # if input gt structure, extract gt contact.
    if (args.gt1 is not None) and (args.gt2 is not None):
        struct1 = Structure(args.gt1)
        struct2 = Structure(args.gt2)
        seq1, _ = struct1.get_seq()
        seq2, _ = struct2.get_seq()
        query1_aligned, seq1_aligned = align_seqs(query, seq1, tmp_dir=f"{tmp_dir}/tmp")
        query2_aligned, seq2_aligned = align_seqs(query, seq2, tmp_dir=f"{tmp_dir}/tmp")
        seq1_mask = [1] * len(seq1_aligned)
        for i, r in enumerate(seq1_aligned):
            if r == "-":
                seq1_mask[i]=0
        seq2_mask = [1] * len(seq2_aligned)
        for i, r in enumerate(seq2_aligned):
            if r == "-":
                seq2_mask[i]=0
        gt1_contact = struct1.pdb_to_contact(seq1_mask)
        gt2_contact = struct2.pdb_to_contact(seq2_mask)
        while query1_aligned.startswith('-'):
            query1_aligned = query1_aligned[1:]
            gt1_contact = gt1_contact[1:, 1:]
        while query1_aligned.endswith('-'):
            query1_aligned = query1_aligned[:-1]
            gt1_contact = gt1_contact[:-1, :-1]
        while query2_aligned.startswith('-'):
            query2_aligned = query2_aligned[1:]
            gt2_contact = gt2_contact[1:, 1:]
        while query2_aligned.endswith('-'):
            query2_aligned = query2_aligned[:-1]
            gt2_contact = gt2_contact[:-1, :-1]
        if ("-" in query1_aligned) or ("-" in query2_aligned):
            raise ValueError(f"Align Error! Query seq contains gaps!\n{query1_aligned}\n{query2_aligned}")
        gt1_contact = torch.tensor(gt1_contact)
        gt2_contact = torch.tensor(gt2_contact)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 3), sharex=True, sharey=True)
        ax = ax.flatten()
        ax[0].imshow(gt1_contact, origin='lower', cmap='Blues')
        ax[1].imshow(gt2_contact, origin='lower', cmap='Blues')
        ax[0].set_title(f"gt1")
        ax[1].set_title(f"gt2")
        plt.savefig(f"{args.o}/gt_contacts.png")
        
        # calculate the matching score, classify sequences
        match_all_gt1_topn = match_score(weight_filter(row_att_all_apc_norm_topn), gt1_contact)
        # match_gt1 = match_score(weight_filter(row_att_apc_diag), gt1_contact)
        match_all_gt2_topn = match_score(weight_filter(row_att_all_apc_norm_topn), gt2_contact)
        # match_gt2 = match_score(weight_filter(row_att_apc_diag), gt2_contact)
        
        gt1_id_topn = np.sort(torch.where((match_all_gt1_topn.sum(-1)>match_all_gt2_topn.sum(-1)))[0].numpy())
        gt2_id_topn = np.sort(torch.where((match_all_gt1_topn.sum(-1)<match_all_gt2_topn.sum(-1)))[0].numpy())
        lprint(f"The number of sequences matching ground truth 1 is {len(gt1_id_topn)}", f)
        lprint(f"The number of sequences matching ground truth 2 is {len(gt2_id_topn)}", f)
        supervised_dir = "supervised_cluster"
        if not os.path.exists(f"{args.o}/{supervised_dir}"):
            os.makedirs(f"{args.o}/{supervised_dir}")
        if 0 not in gt1_id_topn:
            gt1_id_topn = np.insert(gt1_id_topn, 0, 0)
        if 0 not in gt2_id_topn:
            gt2_id_topn = np.insert(gt2_id_topn, 0, 0)
        write_fasta(np.array(data)[gt1_id_topn][:, 0], np.array(data)[gt1_id_topn][:, 1], f"{args.o}/{supervised_dir}/gt1.a3m")
        write_fasta(np.array(data)[gt2_id_topn][:, 0], np.array(data)[gt2_id_topn][:, 1], f"{args.o}/{supervised_dir}/gt2.a3m")
        
    row_att_all_apc_norm_topn_sumhead = weight_filter(row_att_all_apc_norm_topn).sum(1)
    # only consider the upper triangle
    tri_id = np.triu_indices(L, k=1)
    data_r = np.zeros((N, tri_id[0].shape[0]))
    for i in range(row_att_all_apc_norm_topn_sumhead.shape[0]):
        data_r[i] = row_att_all_apc_norm_topn_sumhead[i][tri_id]
        
    if args.cluster_method.upper() == "DBSCAN":
        n_clusters=[]
        eps_test_vals=np.arange(args.min_eps, args.max_eps+args.eps_step, args.eps_step)


        if args.eps_val is None: # performing scan
            lprint('eps\tn_clusters\tn_not_clustered',f)

            for eps in eps_test_vals:
                clustering = DBSCAN(eps=eps, min_samples=args.min_samples).fit(ohe_seqs[1:, :])
                n_clust = len(set(clustering.labels_))
                n_not_clustered = len(clustering.labels_[np.where(clustering.labels_==-1)])
                lprint('%.2f\t%d\t%d' % (eps, n_clust, n_not_clustered),f)
                n_clusters.append(n_clust)
                if eps>10 and n_clust==1:
                    break

            eps_to_select = eps_test_vals[np.argmax(n_clusters)]
        else:
            eps_to_select = args.eps_val

        # perform actual clustering

        clustering = DBSCAN(eps=eps_to_select, min_samples=args.min_samples).fit(data_r[1:])

        lprint('Selected eps=%.2f' % eps_to_select,f)
        
    elif args.cluster_method.upper() == "KMEANS":
        num_cluster = math.floor(N/args.mean_cluster)
        if num_cluster <= 1:
            lprint('MSA (%d) depth is too low to cluster.' % (N),f)
            exit()
        clustering = KMeans(n_clusters=num_cluster).fit(data_r[1:])
    clusters = [x for x in set(clustering.labels_) if x>=0]
    len_unclustered = len(np.where(clustering.labels_==-1)[0])
    lprint('%d clusters, %d seqs not clustered.' % (len(clusters), len_unclustered),f)
    
 
    unsupervised_dir = "unsupervised_cluster"
    if not os.path.exists(f"{args.o}/{unsupervised_dir}"):
        os.makedirs(f"{args.o}/{unsupervised_dir}")
    for clust in clusters:
        id = np.where(clustering.labels_==clust)[0].tolist()
        if 0 not in id:
            id.append(0)
        id = sorted(id)
        write_fasta(np.array(data)[id][:, 0], np.array(data)[id][:, 1], outfile=f"{args.o}/{unsupervised_dir}/{args.keyword}_{clust}.a3m")

    f.close()
    os.system(f"rm -rf {tmp_dir}")