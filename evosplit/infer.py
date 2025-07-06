import torch
from .esm import pretrained
import gc
import math
def msatr(data):
    if torch.cuda.is_available():  
        dev = "cuda" 
    else:  
        dev = "cpu"
    device = torch.device(dev)
    gc.collect()
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated(device=device))
    msa_transformer, msa_alphabet = pretrained.esm_msa1b_t12_100M_UR50S()
    msa_batch_converter = msa_alphabet.get_batch_converter()
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(data)
    msa_transformer = msa_transformer.to(device)
    msa_transformer = msa_transformer.eval()
    msa_batch_tokens = msa_batch_tokens.to(device)
    # msa_transformer.cuda().half()
    with torch.no_grad():
        results = msa_transformer(msa_batch_tokens, repr_layers=[11], return_contacts=True, row_att_all=True)
    # for n,file in enumerate(files):
    msa_transformer.cpu()
    msa_batch_tokens.cpu()
    del msa_transformer
    del msa_batch_tokens
    gc.collect()
    torch.cuda.empty_cache()
    # row_att = results["row_attentions"].cpu()
    return msa_alphabet, results

def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

def map_top(m, top_L=15/2):
    L = m.shape[-1]
    top_n = math.floor(top_L*L)
    filter_id = torch.argsort(m.flatten(), descending=True)[top_n:] # 拉成一维时的索引
    return map_filter(m, filter_id), filter_id

def map_filter(m, filter_id):
    tmp = m.flatten()
    tmp[filter_id]=0
    tmp = tmp.reshape(m.shape[0], m.shape[1])
    return tmp

def weight_filter(map, theta=0):
    tmp = map.float()
    tmp[torch.where(map<theta)]=0
    return tmp

def match_score(m, truth):
    # match_score = torch.div(torch.einsum("abcd->ab", torch.mul(m, truth)), torch.einsum("abcd->ab", m))
    match_score = torch.div(torch.einsum("abcd->ab", torch.mul(m, truth)), torch.sum(truth))
    return match_score