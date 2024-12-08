# _*_ coding:utf-8 _*_
from .dataset import Dataset
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import os
from copy import deepcopy
import logging 
import sys
logging.basicConfig(format="%(asctime)s [%(levelname)s]%(name)s: %(message)s",datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.DEBUG,
            handlers=[logging.StreamHandler(sys.stdout)],)
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class Refiner():
    def __init__(self, iters, tau, lam):
        self.iters = iters
        self.tau = tau
        self.lam = lam
        self.ent_sims = None
        self.rel_sims = None


    def refine_sims(self, dataset:Dataset, target:str="ent", iters:int=0):
        if iters==0:
            logger.info(f"{iters}th Iteration: Initialize {target.upper()} sims ...")
            sims = get_init_sims(dataset, target)
            return sims

        logger.info(f"{iters}th Iteration: Updating {target.upper()} sims ...")    
        if target.lower()=="ent":
            sims = update_sims(self, dataset, iters, target="ent")
        else :
            sims = update_sims(self, dataset, iters, target="rel")
        return sims
        


def update_sims(refiner:Refiner, dataset:Dataset, iters:int, target:str="ent"):
    def get_fin_sim_batch(batch:int):
        fin_sim = []
        norm = []
        fin_sim_len = 0
        for epoch in range(t_node_size//batch+1):
            start_idx = epoch*batch
            end_idx = (epoch+1)*batch
            if start_idx >= t_node_size : break;
            if end_idx > t_node_size : end_idx = t_node_size

            source_idx = np.concatenate([np.repeat(key_cnt[s_id], len(key_cnt[t_id]), axis=0) if t_id in key_cnt else np.repeat(key_cnt[s_id], 0, axis=0) for t_id in t_ids[start_idx:end_idx]]).astype(int)
            target_idx = np.vstack([np.tile(key_cnt[t_id], (len_cnt_s, 1)) if t_id in key_cnt else np.tile([None, None],(0,1)) for t_id in t_ids[start_idx:end_idx]]).astype(int)
            
            main_sim = tf.gather_nd(
                    indices = np.column_stack([source_idx[:,1],target_idx[:,1]]),
                    params  = main_sims
                    )
            sub_sim = tf.gather_nd(
                    indices = np.column_stack([source_idx[:,0],target_idx[:,0]]),
                    params  = sub_sims
                    )
            sub_sim *= reversed_tau
            sub_sim = np.exp(sub_sim)
            cal_mul = np.multiply(sub_sim, main_sim)

            if start_idx == 0 :
                fin_sim_batch = np.array([np.sum(cal_mul[t_cnt_idx[t_idx-1]*len_cnt_s : t_cnt_idx[t_idx]*len_cnt_s]) for t_idx in range(1, end_idx)])
                fin_sim_batch = np.concatenate([[np.sum(cal_mul[0 : t_cnt_idx[0]*len_cnt_s])] if t_cnt_idx[0]>0 else [0], fin_sim_batch])
                norm_batch = [np.sum(sub_sim[t_cnt_idx[t_idx-1]*len_cnt_s : t_cnt_idx[t_idx]*len_cnt_s]) for t_idx in range(1, end_idx)]
                norm_batch = np.concatenate([[np.sum(sub_sim[:t_cnt_idx[0]*len_cnt_s])] if t_cnt_idx[0]>0 else [0], norm_batch])
            
            else :
                fin_sim_batch = np.array([np.sum(cal_mul[t_cnt_idx[t_idx-1]*len_cnt_s-fin_sim_len : t_cnt_idx[t_idx]*len_cnt_s-fin_sim_len]) for t_idx in range(start_idx, end_idx)])
                norm_batch = np.array([np.sum(sub_sim[t_cnt_idx[t_idx-1]*len_cnt_s-fin_sim_len : t_cnt_idx[t_idx]*len_cnt_s-fin_sim_len]) for t_idx in range(start_idx, end_idx)])
            
            fin_sim.extend(fin_sim_batch)
            norm.extend(norm_batch)
            fin_sim_len += len(cal_mul)
            
        fin_sim = tf.math.divide_no_nan(fin_sim, norm)

        return fin_sim


    lam = refiner.lam
    if target=="ent":
        not_target="rel"
        main_sims = refiner.ent_sims
        sub_sims = refiner.rel_sims
        cnt = dataset.cnt[0]

    else :
        not_target="ent"
        main_sims = refiner.rel_sims
        sub_sims =refiner.ent_sims
        cnt = dataset.cnt[1]


    s_node_size, t_node_size = main_sims.shape
    s_ids = dataset.idx2id[target][0]
    t_ids = dataset.idx2id[target][1]
    key_cnt = dict() # key : head_id(KGs), val : list of [relation_idx, tail_idx](sims matrix)
    for h_id in sorted(cnt.keys()):
        if h_id in dataset.id2idx[target][0] : 
            tag = 0
        else :
            tag = 1
        for r_id, t_id in cnt[h_id] :
            r_idx = dataset.id2idx[not_target][tag][r_id]
            t_idx = dataset.id2idx[target][tag][t_id]
            if h_id not in key_cnt : 
                key_cnt[h_id]=[]
            key_cnt[h_id].append([r_idx, t_idx])

    t_cnt_idx = np.zeros(t_node_size, dtype='int32')
    idx = 0
    for t_idx, t_id in enumerate(t_ids):
        if t_id in cnt :
            idx += len(cnt[t_id])
        t_cnt_idx[t_idx] = idx

    reversed_tau = 1/refiner.tau # avoid overflow
    
    fin_sims = []

    # for s_id in tqdm(s_ids):
    #     if s_id not in cnt: # check s_id has triple
    #         fin_sim = np.zeros(t_node_size)
    #         fin_sims.append(fin_sim)
    #         continue;
        
    #     len_cnt_s = len(key_cnt[s_id])
    #     batch = batch_size//len_cnt_s + 16
    #     fin_sim = get_fin_sim_batch(batch)
    #     fin_sims.append(fin_sim)

    # fin_sims = np.array(fin_sims)
    # fin_sims = (lam)*main_sims + (1-lam)*fin_sims
    # return fin_sims

    # if target == "ent" :
    for s_id in tqdm(s_ids):
        if s_id not in key_cnt: # check s_id has triple
            fin_sim = np.zeros(t_node_size)
            fin_sims.append(fin_sim)
            continue
        
        len_cnt_s = len(key_cnt[s_id])

        source_idxs = np.concatenate([np.repeat(key_cnt[s_id], len(key_cnt[t_id]), axis=0) if t_id in key_cnt else np.repeat(key_cnt[s_id], 0, axis=0) for t_id in t_ids]).astype(int)
        target_idxs = np.vstack([np.tile(key_cnt[t_id], (len_cnt_s, 1)) if t_id in key_cnt else np.tile([None, None],(0,1)) for t_id in t_ids]).astype(int)
        
        if source_idxs.shape[0] > 2*10**9:
            b = 10
            while True:
                try:
                    fin_sim = get_fin_sim_batch(b)
                except:
                    b -= 2
                else:
                    break
        elif source_idxs.shape[0] > 10**9:
            fin_sim = get_fin_sim_batch(50)
        elif source_idxs.shape[0] > 10**8:
            fin_sim = get_fin_sim_batch(100)
        else :
            sub_sim = tf.gather_nd(
                    indices = np.column_stack([source_idxs[:,0],target_idxs[:,0]]),
                    params  = sub_sims
                    )
            sub_sim *= reversed_tau
            sub_sim = np.exp(sub_sim)
            # sub_sim = tf.math.exp(sub_sim)
            main_sim = tf.gather_nd(
                    indices = np.column_stack([source_idxs[:,1],target_idxs[:,1]]),
                    params  = main_sims
                    )

            cal_mul = np.multiply(sub_sim, main_sim)
            fin_sim = np.array([np.sum(cal_mul[t_cnt_idx[t_idx-1]*len_cnt_s : t_cnt_idx[t_idx]*len_cnt_s]) for t_idx in range(1, t_node_size)])
            fin_sim = np.concatenate([[np.sum(cal_mul[0 : t_cnt_idx[0]*len_cnt_s])] if t_cnt_idx[0]>0 else [0] , fin_sim])
            
            norm = [np.sum(sub_sim[t_cnt_idx[t_idx-1]*len_cnt_s : t_cnt_idx[t_idx]*len_cnt_s]) for t_idx in range(1, t_node_size)]
            norm = np.concatenate([[np.sum(sub_sim[:t_cnt_idx[0]*len_cnt_s])] if t_cnt_idx[0]>0 else [0] , norm])

            fin_sim = tf.math.divide_no_nan(fin_sim, norm)
        

        fin_sims.append(fin_sim)

    # else :
    #     for s_id in tqdm(s_ids):
    #         if s_id not in cnt: # check s_id has triple
    #             fin_sim = np.zeros(t_node_size)
    #             fin_sims.append(fin_sim)
    #             continue;
            
    #         len_cnt_s = len(key_cnt[s_id])
    #         batch = batch_size//len_cnt_s + 16
    #         fin_sim = get_fin_sim_batch(batch)
    #         fin_sims.append(fin_sim)

    fin_sims = np.array(fin_sims)
    fin_sims = (lam)*main_sims + (1-lam)*fin_sims
    return fin_sims
   


def get_init_sims(dataset:Dataset, target="ent", depth=2):
    def cal_sims(s_ent_ids, t_ent_ids,feature):
        feature_a = tf.gather(indices=s_ent_ids , params=feature)
        feature_b = tf.gather(indices=t_ent_ids , params=feature)
        return tf.matmul(feature_a,tf.transpose(feature_b,[1,0]))
    
    save_path = dataset.data_path+f"init_score"
    if target=="ent" and os.path.exists(f"{save_path}.npy"):
        logger.info(f"Load initial align score \'{save_path}.npy\' ...")
        sims = np.load(f"{save_path}.npy")
        logger.info(f"File loaded ...")
        return sims
    
    if target=="ent" : tag = 0
    else : tag = 1

    s_ids = dataset.idx2id[target][0]
    t_ids = dataset.idx2id[target][1]
    feature = dataset.feature[tag]
    adj_mat = dataset.adj_mat[tag]

    sims = cal_sims(s_ids, t_ids, feature)
    for _ in range(depth):    
        feature = tf.sparse.sparse_dense_matmul(adj_mat, feature)
        feature = tf.nn.l2_normalize(feature,axis=-1)
        sims += cal_sims(s_ids, t_ids,feature)
    
    sims /= depth+1
    sims = sinkhorn(sims)
    if target == "ent":
        logger.info(f"Save initial align score as \'{save_path}\' ...")
        np.save(save_path, sims)
        logger.info("File saved ...")
        
    return sims

def sinkhorn(sims, iters=10):
        sinkhorn_sims = tf.exp(sims*50)
        for _ in range(iters):
            sinkhorn_sims = sinkhorn_sims / tf.reduce_sum(sinkhorn_sims,axis=1,keepdims=True)
            sinkhorn_sims = sinkhorn_sims / tf.reduce_sum(sinkhorn_sims,axis=0,keepdims=True)
        return sinkhorn_sims
