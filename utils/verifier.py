# _*_ coding:utf-8 _*_
from .dataset import Dataset
from .refiner import Refiner, sinkhorn, get_init_sims
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from copy import deepcopy
from sentence_transformers import SentenceTransformer, util
import logging 
import sys
logging.basicConfig(format="%(asctime)s [%(levelname)s]%(name)s: %(message)s",datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.DEBUG,
            handlers=[logging.StreamHandler(sys.stdout)],)
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class Verifier():
    def __init__(self, dataset:Dataset, refiner:Refiner, model:str='sentence-transformers/all-MiniLM-L6-v2'):
        logger.info(f"Setting verifier ...")

        ent_sims0 = get_init_sims(dataset)
        ent_sims2 = sinkhorn(refiner.ent_sims)
        rel_sims2 = sinkhorn(refiner.rel_sims)
        trans_ent2 = np.transpose(ent_sims2)
        trans_rel2 = np.transpose(rel_sims2)

        ranks0, sort_sims0, results0, log_cont = eval_sims(ent_sims0, dataset, verify=True, log=False)
        logger.debug(f"init entity > {log_cont}")
        ranks2, sort_sims2, results2, log_cont = eval_sims(ent_sims2, dataset, verify=True, log=False)
        logger.debug(f"entity      > {log_cont}")
        trans_ranks2, trans_sort_sims2, trans_results2, log_cont = eval_sims(trans_ent2, dataset, trans=True, verify=True, log=False)
        logger.debug(f"trans entity> {log_cont}")
        consistency = get_consistency(ent_sims0, ent_sims2)
        confidence = get_confidence(sort_sims2)

        self.model_sts = SentenceTransformer(model)
        self.ranks = [ranks0, ranks2, trans_ranks2]
        self.results = [results0, results2, trans_results2]
        self.consistency = consistency
        self.confidence = confidence
        self.ent_sims2 = ent_sims2
        self.rel_sims2 = rel_sims2
        self.trans_ent2 = trans_ent2
        self.trans_rel2 = trans_rel2

    def get_premise_hypothesis(self, dataset:Dataset, s_idx:int, topk:int=20, trans:bool=False, sentencek:int=10):
        if trans: 
            ranks2 = self.ranks[2]
            ent_sims, rel_sims = self.trans_ent2, self.trans_rel2
            tag = (1,0)
        else: 
            ent_sims, rel_sims = self.ent_sims2, self.rel_sims2
            ranks2 = self.ranks[1]
            tag = (0,1)
        ent_names, rel_names = dataset.names["ent"], dataset.names["rel"]
        cnt = dataset.cnt[0]
        rank_list = ranks2[s_idx,:topk]

        s_id = dataset.idx2id["ent"][tag[0]][s_idx]
        t_ids = dataset.idx2id["ent"][tag[1]]
        t_rank_ids = np.take(t_ids, rank_list)

        premise_idx_list = []
        len_cnt_s_id = len(cnt[s_id])
        s_tail_cnt = np.zeros(len_cnt_s_id)
        for i in range(len_cnt_s_id):
            r_id, t_id = cnt[s_id][i] 
            r_idx = dataset.id2idx["rel"][tag[0]][r_id]
            t_idx = dataset.id2idx["ent"][tag[0]][t_id]
            premise_idx_list.append([r_idx, t_idx])
            if t_id in cnt : 
                s_tail_cnt[i] += len(cnt[t_id])
        s_tail_cnt = np.argsort(-s_tail_cnt)[:sentencek] # Frequency 
        premise_idx_list = np.take(premise_idx_list, s_tail_cnt, axis=0)

        premise = ent_names[s_id]

        if len_cnt_s_id>0:
            premise += ", which"
            for i in s_tail_cnt:
                r_id, t_id = cnt[s_id][i]
                premise += f" {rel_names[r_id]} is {ent_names[t_id]},"
            premise = premise[:-1] + "."
        hypothesis_list = []

        for t_head_id in t_rank_ids :
            hypothesis = ent_names[t_head_id]

            hypothesis += ", which"
            
            hypo_idx_list = []
            t_rel_dict = {}
            for r_id, t_id in cnt[t_head_id]:
                r_idx = dataset.id2idx["rel"][tag[1]][r_id]
                t_idx = dataset.id2idx["ent"][tag[1]][t_id]
                hypo_idx_list.append([r_idx, t_idx])
                if r_idx not in t_rel_dict: 
                    t_rel_dict[r_idx] = set()
                t_rel_dict[r_idx].add(t_idx)
            hypo_idx_list = np.array(hypo_idx_list)

            for sr_idx, st_idx in premise_idx_list:
                hypo_rel_sims = np.take(rel_sims[sr_idx], hypo_idx_list[:, 0])
                hypo_rel_ranks = np.take(hypo_idx_list[:, 0] ,np.argsort(hypo_rel_sims))
                best_rel_idx = hypo_rel_ranks[-1]
                best_rel_id = dataset.idx2id["rel"][tag[1]][best_rel_idx]
        
                tail_list = sorted(list(t_rel_dict[best_rel_idx]))
                target_tail_sims = np.take(ent_sims[st_idx], tail_list)
                target_tail_ranks = np.take(tail_list, np.argsort(target_tail_sims))
                best_tail_idx = target_tail_ranks[-1]
                best_tail_id = dataset.idx2id["ent"][tag[1]][best_tail_idx]
                hypothesis += f" {rel_names[best_rel_id]} is {ent_names[best_tail_id]},"
            
            hypothesis = hypothesis[:-1] + "."
            hypothesis_list.append(hypothesis)
        return premise, hypothesis_list

    def verify(self, dataset:Dataset, p1=20, p2=20, topk=20, sentencek=10):
        model_sts = self.model_sts
        ranks2   = self.ranks[1]
        results2 = self.results[1]
        trans_ranks2 = self.ranks[2]
        confidence = self.confidence
        consistency = self.consistency
        test_pair = dataset.aligned_pairs

        confidence_threshold = np.percentile(confidence,p1)
        consistency_threshold = np.percentile(consistency,p2)
        logger.info(f"Threshold: confidence-{p1}%({confidence_threshold:.4f}), consistency-{p2}%({consistency_threshold:.4f})")
        
        verify_ranks = []
        verified_idx = []
        append_count = 0
        reject_count = 0
        for i, s_id in enumerate(tqdm(test_pair[:,0])):
            s_idx = dataset.id2idx["ent"][0][s_id]
            rank_list = ranks2[s_idx,:topk]
            if results2[i][1]>topk-1 :
                verify_ranks.append(rank_list)
                continue
            if (confidence[s_idx]>confidence_threshold and consistency[s_idx]>consistency_threshold) :
                verify_ranks.append(rank_list)
                continue

            s_premise, s_hypothesis_list = self.get_premise_hypothesis(s_idx, topk=topk, sentencek=sentencek)
            s_rank = sts(model_sts, s_premise, s_hypothesis_list)
            top1_t_idx=ranks2[s_idx, s_rank[0]]
            t_premise, t_hypothesis_list = self.get_premise_hypothesis(top1_t_idx, topk=topk, trans=True, sentencek=sentencek)
            t_rank = sts(model_sts, t_premise, t_hypothesis_list)
            top1_s_idx=trans_ranks2[top1_t_idx, t_rank[0]]
            if top1_s_idx != i :
                verify_ranks.append(rank_list)
                reject_count += 1
                continue
            s_rank = np.take(a=rank_list, indices=s_rank)
            verify_ranks.append(s_rank)
            append_count += 1
            verified_idx.append(i)

        verify_ranks = np.array(verify_ranks)
        logger.info(f"append count: {append_count}")
        logger.info(f"reject count: {reject_count}")

        return verify_ranks, verified_idx


def cal(results):
    hits1,hits10,mrr = 0,0,0
    for x in results[:,1]:
        if x < 1:
            hits1 += 1
        if x < 10:
            hits10 += 1
        mrr += 1/(x + 1)
    return hits1,hits10,mrr

def eval_sims(sims, data:Dataset, trans=False, verify=False, batch_size=1024, log=True):   
    ans = {}
    ans_keys = []
    test_pair = data.aligned_pairs
    # to match id(KGs) - idx(sim matrix)
    for s_id, t_id in test_pair: 
        if trans:
            ans[data.id2idx["ent"][1][t_id]] = data.id2idx["ent"][0][s_id]
            ans_keys.append(data.id2idx["ent"][1][t_id])
        else:
            ans[data.id2idx["ent"][0][s_id]] = data.id2idx["ent"][1][t_id]
            ans_keys.append(data.id2idx["ent"][0][s_id])

    results = []
    ranks = []
    sort_sims = []
    if verify:
        for epoch in range(len(sims)//batch_size+1) :
            sim = sims[epoch*batch_size : (epoch+1)*batch_size]
            rank = tf.argsort(-sim,axis=-1)        
            ranks.append(rank)
        ranks = np.concatenate(ranks,axis=0)
        for i in range(len(sims)):
            sort_sims.append(np.take(sims[i], ranks[i]))
        sort_sims = np.array(sort_sims)

        for epoch in range(len(test_pair)//batch_size+1) :
            test_pair_rank = tf.gather(indices=ans_keys[epoch*batch_size : (epoch+1)*batch_size], params=ranks)
            ans_rank = np.array([ans[ans_keys[i]] for i in range(epoch * batch_size, min((epoch+1) * batch_size, len(ans_keys)))])
            results.append(tf.where(tf.equal(tf.cast(test_pair_rank, ans_rank.dtype), tf.tile(np.expand_dims(ans_rank,axis=1), [1,len(sims[0])]))).numpy())
        
    else:
        for epoch in range(len(test_pair)//batch_size+1) :
            sim = tf.gather(indices=ans_keys[epoch*batch_size : (epoch+1)*batch_size], params=sims)
            test_pair_rank = tf.argsort(-sim,axis=-1)
            ans_rank = np.array([ans[ans_keys[i]] for i in range(epoch * batch_size, min((epoch+1) * batch_size, len(ans_keys)))])
            results.append(tf.where(tf.equal(tf.cast(test_pair_rank, ans_rank.dtype), tf.tile(np.expand_dims(ans_rank,axis=1), [1,len(sims[0])]))).numpy())
    
    results = np.concatenate(results,axis=0)
    hits1,hits10,mrr = cal(results)
    log_cont = "hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (hits1/len(results)*100,hits10/len(results)*100,mrr/len(results)*100)
    if log:
        logger.info(log_cont)
        return ranks, sort_sims, results
    else :
        return ranks, sort_sims, results, log_cont
    
def new_rank_results (verifier:Verifier, dataset:Dataset, verify_ranks, batch_size=1024):
    ranks = verifier.ranks[1]
    ent_sims = verifier.ent_sims2
    results = verifier.results[1]
    test_pair = dataset.aligned_pairs
    ans = {}
    ans_keys = []
    for s_id, t_id in test_pair: 
        s_idx = dataset.id2idx["ent"][0][s_id]
        t_idx = dataset.id2idx["ent"][1][t_id]
        ans[s_idx] = t_idx
        ans_keys.append(s_idx)

    new_ranks = deepcopy(ranks)
    verify_ranks = verify_ranks.astype(int)
    for i, s_idx in enumerate(ans_keys):
        new_ranks[s_idx] =  np.concatenate([verify_ranks[i], new_ranks[s_idx, len(verify_ranks[0]):]])
    new_ranks = np.array(new_ranks)

    new_results = []
    for epoch in range(len(test_pair)//batch_size+1) :
        rank = tf.gather(indices=ans_keys[epoch*batch_size : (epoch+1)*batch_size], params=new_ranks)
        ans_rank = np.array([ans[ans_keys[i]] for i in range(epoch * batch_size, min((epoch+1) * batch_size, len(ans_keys)))])
        new_results.append(tf.where(tf.equal(tf.cast(rank, ans_rank.dtype), tf.tile(np.expand_dims(ans_rank,axis=1), [1,len(ent_sims[0])]))).numpy())
    new_results = np.concatenate(new_results,axis=0)
    hits1,hits10,mrr = cal(new_results)
    log_cont = f"Verify result > hits@1 : {hits1/len(new_results)*100 :.2f} hits@10 : {hits10/len(new_results)*100 :.2f} MRR : {mrr/len(new_results)*100 :.2f}"
    logger.info(log_cont)
    return new_ranks, new_results
        
def verified_dist(verifier:Verifier, new_results, verified_idx):
    results = verifier.results[1]
    verified_case_keys = ["o->o", "o->x", "x->o", "x->x"]
    verified_case = dict()
    for key in verified_case_keys:
        verified_case[key] = []
    changed=[0,0,0]
    for test_pair_idx in verified_idx:
        before = results[test_pair_idx][1]
        after = new_results[test_pair_idx][1]
        if before-after > 0 :changed[0] += 1
        elif before-after <0 : changed[-1] +=1
        else : changed[1]+= 1

        item = [test_pair_idx, int(before), int(after)]
        if before>0:
            if after>0: verified_case[verified_case_keys[3]].append(item)
            else: verified_case[verified_case_keys[2]].append(item)
        else : 
            if after>0: verified_case[verified_case_keys[1]].append(item)
            else: verified_case[verified_case_keys[0]].append(item)
    logger.info(f"better : {changed[0]}, worse : {changed[-1]}, non-changed : {changed[1]}")
    for i in verified_case_keys:
        logger.info(f"{i} : {len(verified_case[i])} ")

    return verified_case


    

def get_consistency(sims0, sims2):
    cos_sims = []
    for sim0, sim2 in zip(sims0, sims2):
        cos_sim = np.dot(sim0, sim2)/(np.linalg.norm(sim0)*np.linalg.norm(sim2))
        cos_sims.append(cos_sim)
    cos_sims = np.array(cos_sims)
    consistency = cos_sims ** 3
    return consistency

def get_confidence(sort_sims2):
    return sort_sims2[:,0]-sort_sims2[:,1]

def sts(model, premise, hypothesis_list) :
    premise_emb = model.encode(premise)
    hypo_sim_list = []
    for hypo in hypothesis_list:
        hypo_sim_list.append(util.pytorch_cos_sim(premise_emb, model.encode(hypo))[0][0])
    sim_rank = np.argsort(np.array(hypo_sim_list)*-1)
    return sim_rank
