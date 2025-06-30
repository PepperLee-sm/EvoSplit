import argparse
import torch
from scripts.utils import QID_filter, load_fasta, write_fasta, lprint
from scripts.structure import Structure
from scripts.infer import msatr, apc, symmetrize, map_top, map_filter, match_score, weight_filter
from scripts.align import align_seqs
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
import math
from sklearn.cluster import KMeans


if __name__=='__main__':

    p = argparse.ArgumentParser(description=
    """
    Clustering sequences by separating coevolutionary signals belonging to different conformations.
    Lishimian, 2024
    """)

    p.add_argument("keyword", action="store", help="Keyword to call all generated MSAs.")
    p.add_argument("-i", action='store', help='fasta/a3m file of original alignment.')
    p.add_argument("-o", action="store", help='name of output directory to write MSAs to.')
    p.add_argument("--gap_cutoff", action='store', type=float, default=0.25, help='Remove sequences with gaps representing more than this frac of seq.')
    p.add_argument("--subfamily_MSA_depth", action='store', type=int, default=1024, help='Depth of subfamily MSA.')
    p.add_argument("-gt1", action='store', default=None, help='pdb file of known ground truth conformation.')
    p.add_argument("-gt2", action='store', default=None, help='pdb file of known ground truth conformation.')
    p.add_argument("--topL", action='store', type=float, default=15/2, help='Number of coevolved amino acid pairs (@L) retained to be analysed. If =7.5, 7.5L pairs.')
    p.add_argument("--nstep", action='store', type=int, default=5, help="Iterations number of supervised clustering")
    p.add_argument("--ts1", action='store', default="mean", help="Iterations number of supervised clustering")
    p.add_argument("--ts2", action='store', type=float, default=0.25, help="Iterations number of supervised clustering")
    
    
    args = p.parse_args()

    
    os.makedirs(args.o, exist_ok=True)
    # 读取msa，按gap cutoff过滤
    f = open(os.path.join(args.o, "%s.log"% args.keyword), 'w')
    IDs, seqs = load_fasta(args.i, f, args.gap_cutoff)
    
    query = seqs[0]
    
    tmp_dir = os.path.join(args.o, "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    # 判断输入msa的深度是否超过1024，若超过则用qid进行过滤
    L = len(seqs[0])
    N = len(seqs)
    if N > args.subfamily_MSA_depth:
        seqs, IDs = QID_filter(seqs, IDs, num_seqs=args.subfamily_MSA_depth, tmp_dir=tmp_dir)
        N = len(seqs)
    lprint(f"The length of query sequence is {L}.\nThe depth of MSA to be analysed is {N}.", f)
    # 将待分析的所有msa写入fasta文件
    write_fasta(IDs, seqs, outfile=os.path.join(args.o, f"{args.keyword}.a3m"))

    data = list(zip(IDs, seqs))
    alphabet, results = msatr(data)
    contacts = results['contacts'].cpu()[0]
    contacts_ = contacts.clone()
    print(args.topL)
    contacts_filtered, filter_id = map_top(contacts_, args.topL)
    top_contacts = contacts_filtered.numpy()
    top_contacts[np.where(top_contacts>0)] = 1
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 3), sharex=True, sharey=True)
    ax = ax.flatten()
    ax[0].imshow(contacts, origin='lower', cmap='Blues')
    ax[1].imshow(top_contacts, origin='lower', cmap='Blues')
    ax[0].set_title("prob")
    ax[1].set_title("contacts (top 15/2L)")
    plt.savefig(f"{args.o}/msatr_contact.png")

    row_att = results['row_attentions'].cpu()[:, -1, :, :, :]
    
    row_att_all = results['row_attentions_all'].cpu()
    if alphabet.prepend_bos:
        row_att_all = row_att_all[:, 0, :, 1:, 1:]
        row_att = row_att[..., 1:, 1:]
        
    # 做对称化和apc
    row_att_apc = apc(symmetrize(row_att))
    row_att_all_apc = apc(symmetrize(row_att_all))
    # 对角线元素设为0
    diag_m = torch.ones((L, L))-torch.eye(L)
    row_att_apc_diag = torch.mul(row_att_apc, diag_m)
    row_att_all_apc_diag = torch.mul(row_att_all_apc, diag_m)
    # 提取最后一层row attn中topL的位置
    row_att_apc_sum = torch.einsum('abcd->cd', row_att_apc_diag)
    row_att_apc_sum_ = row_att_apc_sum.clone()
    row_att_apc_sum_filtered, filter_id = map_top(row_att_apc_sum_, args.topL)
    
    s, h = row_att_all_apc_diag.shape[0:2]
    row_att_all_apc_norm_topn = torch.zeros_like(row_att_all_apc_diag)
    for i in range(s):
        for j in range(h):
            row_att_all_apc_norm_topn[i, j] = map_filter(row_att_all_apc_diag[i, j], filter_id)
            
    # 若输入gt结构，提取gt contact。
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
        
        data_filt = data
        row_att_all_apc_norm_topn_filt = row_att_all_apc_norm_topn.clone()
        seq_gt1 = []
        seq_gt2 = []
        match_all_gt1_topn_all = []
        match_all_gt2_topn_all = []
        contacts_filt_all = []
        top_contacts_filt_all = []

        for i in range(args.nstep):
            if len(data_filt) < 512:
                break
            # 计算匹配度，分类序列
            match_all_gt1_topn = match_score(weight_filter(row_att_all_apc_norm_topn_filt), gt1_contact)
            match_all_gt2_topn = match_score(weight_filter(row_att_all_apc_norm_topn_filt), gt2_contact)
            gt1_id_topn = torch.where((match_all_gt1_topn.sum(-1)>match_all_gt2_topn.sum(-1)))[0].numpy()
            gt2_id_topn = torch.where((match_all_gt1_topn.sum(-1)<match_all_gt2_topn.sum(-1)))[0].numpy()
            lprint(f"Iteration {i+1}: The number of sequences matching ground truth 1 is {len(gt1_id_topn)}", f)
            lprint(f"Iteration {i+1}: The number of sequences matching ground truth 2 is {len(gt2_id_topn)}", f)
            if args.ts1 == "mean":
                threshold1 = torch.cat([match_all_gt1_topn.sum(-1), match_all_gt2_topn.sum(-1)]).mean()
            else:
                threshold1 = 0
            threshold2 = torch.abs(match_all_gt1_topn.sum(-1)-match_all_gt2_topn.sum(-1)).max()*args.ts2
            
            match_all_gt1_topn_all.append(match_all_gt1_topn.sum(-1))
            match_all_gt2_topn_all.append(match_all_gt2_topn.sum(-1))
            # 可视化匹配度
            fig = sns.jointplot(x=match_all_gt1_topn.sum(-1), y=match_all_gt2_topn.sum(-1), kind="kde", fill=True, height=4)
            limit = max(match_all_gt1_topn.sum(-1).max(), match_all_gt2_topn.sum(-1).max())
            ax = fig.figure.axes
            plt.xlabel('match ground (all)')
            plt.ylabel('match FS (all)')
            plt.xlim(0, limit*1.5)
            plt.ylim(0, limit*1.5)
            plt.plot([0, limit*1.5], [0, limit*1.5], ls="--", c=".3")
            if threshold1 > 0:
                plt.plot([0, limit*1.5], [threshold1, threshold1], ls="--", c="r",)
                plt.plot([threshold1, threshold1], [0, limit*1.5], ls="--", c="r",)
            plt.plot([0, limit*1.5], [threshold2, limit*1.5+threshold2], ls="--", c="r",)
            plt.plot([0, limit*1.5], [-threshold2, limit*1.5-threshold2], ls="--", c="r",)
            plt.savefig(f"{args.o}/match_score_{i}.png")
            gt1_id_topn_filt = torch.where(((match_all_gt1_topn.sum(-1)-match_all_gt2_topn.sum(-1))>=threshold2) & (match_all_gt1_topn.sum(-1)>=threshold1))[0].numpy()
            gt2_id_topn_filt = torch.where(((match_all_gt2_topn.sum(-1)-match_all_gt1_topn.sum(-1))>=threshold2) & (match_all_gt2_topn.sum(-1)>=threshold1))[0].numpy()
            lprint(f"Iteration {i+1}: The number of sequences assigned to ground truth 1 is {len(gt1_id_topn_filt)}", f)
            lprint(f"Iteration {i+1}: The number of sequences assigned to ground truth 2 is {len(gt2_id_topn_filt)}", f)
            seq_gt1 += [data_filt[i] for i in gt1_id_topn_filt]
            seq_gt2 += [data_filt[i] for i in gt2_id_topn_filt]
            
            id_rest = list(set([i for i in range(len(data_filt))])^set(gt1_id_topn_filt)^set(gt2_id_topn_filt))
            data_filt = [data_filt[i] for i in id_rest]
            
            alphabet_filt, results_filt = msatr(data_filt)
            contacts_filt = results_filt['contacts'].cpu()[0]
            contacts_filt_ = contacts_filt.clone()
            contacts_filtered_filt, filter_id_filt = map_top(contacts_filt_)
            top_contacts_filt = contacts_filtered_filt
            top_contacts_filt[np.where(top_contacts_filt>0)] = 1
            
            contacts_filt_all.append(contacts_filt)
            top_contacts_filt_all.append(top_contacts_filt)
            
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 3), sharex=True, sharey=True)
            ax = ax.flatten()
            ax[0].imshow(contacts_filt, origin='lower', cmap='Blues')
            ax[1].imshow(top_contacts_filt, origin='lower', cmap='Blues')
            ax[0].set_title("prob")
            ax[1].set_title("contacts (top 15/2L)")
            plt.savefig(f"{args.o}/msatr_contact_{i}.png")
            row_att_filt = results_filt['row_attentions'].cpu()[:, -1, :, :, :]
            row_att_all_filt = results_filt['row_attentions_all'].cpu()
            if alphabet_filt.prepend_bos:
                row_att_all_filt = row_att_all_filt[:, 0, :, 1:, 1:]
                row_att_filt = row_att_filt[..., 1:, 1:]
            
            # 做对称化和apc
            row_att_apc_filt = apc(symmetrize(row_att_filt))
            row_att_all_apc_filt = apc(symmetrize(row_att_all_filt))
            # 对角线元素设为0
            diag_m = torch.ones((L, L))-torch.eye(L)
            row_att_apc_diag_filt = torch.mul(row_att_apc_filt, diag_m)
            row_att_all_apc_diag_filt = torch.mul(row_att_all_apc_filt, diag_m)
            # 提取最后一层row attn中topL的位置
            row_att_apc_sum_filt = torch.einsum('abcd->cd', row_att_apc_diag_filt)
            row_att_apc_sum_filt_ = row_att_apc_sum_filt.clone()
            row_att_apc_sum_filt_filtered, filter_id = map_top(row_att_apc_sum_filt_, args.topL)
            
            s, h = row_att_all_apc_diag_filt.shape[0:2]
            row_att_all_apc_norm_topn_filt = torch.zeros_like(row_att_all_apc_diag_filt)
            for i in range(s):
                for j in range(h):
                    row_att_all_apc_norm_topn_filt[i, j] = map_filter(row_att_all_apc_diag_filt[i, j], filter_id_filt)
        
        
        match_all_gt1_topn = match_score(weight_filter(row_att_all_apc_norm_topn_filt), gt1_contact)
        match_all_gt2_topn = match_score(weight_filter(row_att_all_apc_norm_topn_filt), gt2_contact)
        gt1_id_topn = torch.where((match_all_gt1_topn.sum(-1)>match_all_gt2_topn.sum(-1)))[0].numpy()
        gt2_id_topn = torch.where((match_all_gt1_topn.sum(-1)<match_all_gt2_topn.sum(-1)))[0].numpy()

        match_all_gt1_topn_all.append(match_all_gt1_topn.sum(-1))
        match_all_gt2_topn_all.append(match_all_gt2_topn.sum(-1))
        lprint(f"Last Iteration: The number of sequences assigned to ground truth 1 is {len(gt1_id_topn)}", f)
        lprint(f"Last Iteration: The number of sequences assigned to ground truth 2 is {len(gt2_id_topn)}", f)
        with open(f"{args.o}/match_score_all_gt1.npy", "wb") as f:
            np.save(f, match_all_gt1_topn_all)
        with open(f"{args.o}/match_score_all_gt2.npy", "wb") as f:  
            np.save(f, match_all_gt2_topn_all)
        with open(f"{args.o}/contacts_all_gt.npy", "wb") as f:
            np.save(f, contacts_filt_all)
        with open(f"{args.o}/top_contacts_all_gt.npy", "wb") as f:
            np.save(f, top_contacts_filt_all)   
            
        seq_gt1_all = [data_filt[i] for i in gt1_id_topn]+seq_gt1
        seq_gt2_all = [data_filt[i] for i in gt2_id_topn]+seq_gt2
            
        lprint(f"The number of all sequences assigned to ground truth 1 is {len(seq_gt1_all)}", f)
        lprint(f"The number of all sequences assigned to ground truth 2 is {len(seq_gt2_all)}", f)
        
        # 分别写入a3m文件
        supervised_dir = "supervised_cluster"
        if not os.path.exists(f"{args.o}/{supervised_dir}"):
            os.makedirs(f"{args.o}/{supervised_dir}")
        write_fasta([i[0] for i in seq_gt1], [i[1] for i in seq_gt1], f"{args.o}/{supervised_dir}/gt1.a3m")
        write_fasta([i[0] for i in seq_gt2], [i[1] for i in seq_gt2], f"{args.o}/{supervised_dir}/gt2.a3m")
        write_fasta([i[0] for i in seq_gt1_all], [i[1] for i in seq_gt1_all], f"{args.o}/{supervised_dir}/gt1_all.a3m")
        write_fasta([i[0] for i in seq_gt2_all], [i[1] for i in seq_gt2_all], f"{args.o}/{supervised_dir}/gt2_all.a3m")
        # TODO: 把ref加进去
        supervised_dir_af2 = "supervised_cluster_af2"
        if not os.path.exists(f"{args.o}/{supervised_dir_af2}"):
            os.makedirs(f"{args.o}/{supervised_dir_af2}")
        if data[0] not in seq_gt1:
            seq_gt1.insert(0,data[0])
        else:
            seq_gt1.remove(data[0])
            seq_gt1.insert(0,data[0])
        if data[0] not in seq_gt2:
            seq_gt2.insert(0,data[0])
        else:
            seq_gt2.remove(data[0])
            seq_gt2.insert(0,data[0])
        if data[0] not in seq_gt1_all:
            seq_gt1_all.insert(0,data[0])
        else:
            seq_gt1_all.remove(data[0])
            seq_gt1_all.insert(0,data[0])
        if data[0] not in seq_gt2_all:
            seq_gt2_all.insert(0,data[0])
        else:
            seq_gt2_all.remove(data[0])
            seq_gt2_all.insert(0,data[0])
        write_fasta([i[0] for i in seq_gt1], [i[1] for i in seq_gt1], f"{args.o}/{supervised_dir_af2}/gt1.a3m")
        write_fasta([i[0] for i in seq_gt2], [i[1] for i in seq_gt2], f"{args.o}/{supervised_dir_af2}/gt2.a3m")
        write_fasta([i[0] for i in seq_gt1_all], [i[1] for i in seq_gt1_all], f"{args.o}/{supervised_dir_af2}/gt1_all.a3m")
        write_fasta([i[0] for i in seq_gt2_all], [i[1] for i in seq_gt2_all], f"{args.o}/{supervised_dir_af2}/gt2_all.a3m")

    # # NMF降维后做无监督聚类
    # row_att_all_apc_norm_topn_sumhead = weight_filter(row_att_all_apc_norm_topn).sum(1)
    # B, L = row_att_all_apc_norm_topn_sumhead.shape[0:2]
    # row_att_all_apc_norm_topn_sumhead = row_att_all_apc_norm_topn_sumhead.view(B, -1)
    # n_components = args.ncomponents
    # nmf = NMF(n_components=n_components,init='nndsvda',tol=5e-3)
    # data_r = nmf.fit_transform(row_att_all_apc_norm_topn_sumhead)
    # comps = nmf.fit(row_att_all_apc_norm_topn_sumhead).components_
    # nrows = 4
    # ncols = math.ceil(n_components/nrows)
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2*ncols, 8), sharex=True, sharey=True)
    # ax = ax.flatten()
    # for i, comp in enumerate(comps):
    #     # vmax = max(comp.max(), -comp.min())
    #     ax[i].imshow(comp.reshape(L, L), origin='lower', cmap='Blues')
    # plt.savefig(f"{args.o}/NMF_{1/4}L.png")
    # clustering = KMeans(n_clusters=math.floor(N/args.mean_cluster)).fit(data_r[1:])
    # clusters = [x for x in set(clustering.labels_) if x>=0]
    # unsupervised_dir = "unsupervised_cluster"
    # if not os.path.exists(f"{args.o}/{unsupervised_dir}"):
    #     os.makedirs(f"{args.o}/{unsupervised_dir}")
    # for clust in clusters:
    #     id = np.where(clustering.labels_==clust)[0].tolist()
    #     if 0 not in id:
    #         id.append(0)
    #     id = sorted(id)
    #     write_fasta(np.array(data)[id][:, 0], np.array(data)[id][:, 1], outfile=f"{args.o}/{unsupervised_dir}/{args.keyword}_{clust}.a3m")

    f.close()