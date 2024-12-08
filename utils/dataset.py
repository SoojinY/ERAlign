# _*_ coding:utf-8 _*_
import numpy as np
import tensorflow as tf
import torch
import os
import logging 
import sys
logging.basicConfig(format="%(asctime)s [%(levelname)s]%(name)s: %(message)s",datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.DEBUG,
            handlers=[logging.StreamHandler(sys.stdout)],)
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class Dataset() : 
    def __init__(self, data_path, mode):
        self.data_path = data_path
        self.mode = mode
        self.id2idx = {"ent": [dict(), dict()], "rel":[dict(), dict()]}
        self.idx2id = {"ent":[[],[]], "rel":[[],[]]} 
        self.names = {"ent": dict(), "rel" : dict()}
        self.set_keys()
        

        logger.info(f"Get Dataset from \'{data_path}\' ... ")
        
        #load KGs and test set
        all_triples,node_size,rel_size = load_triples(data_path)
        logger.info(f"# all triples : {len(all_triples)}\t # node : {node_size}\t # edge : {rel_size}")
        dual_triples, dual_node_size, dual_rel_size = triples_to_dual_triples(all_triples)
        logger.info(f"# dual triples : {len(dual_triples)}\t # dual node : {dual_node_size}\t # dual rel : {dual_rel_size}")
        aligned_pairs = load_aligned_pair(data_path)
        logger.info(f"# test pairs : {len(aligned_pairs)}")

        # dict: key - head_id, val - [rel_id, tail_id]
        cnt = get_cnt(all_triples) 
        dual_cnt = get_cnt(dual_triples)

        adj_mat = get_adjacency_matrix(all_triples, node_size)
        adj_mat_dual = get_adjacency_matrix(dual_triples, rel_size)
        feature = self.get_feature(mode, "ent")
        feature_dual = self.get_feature(mode, "rel")

        self.triples = [all_triples, dual_triples]
        self.aligned_pairs = aligned_pairs
        self.cnt = [cnt, dual_cnt]
        self.adj_mat = [adj_mat, adj_mat_dual]
        self.feature = [feature, feature_dual]

        s_ent_ids = self.idx2id["ent"][0]
        t_ent_ids = self.idx2id["ent"][1]
        s_rel_ids = self.idx2id["rel"][0]
        t_rel_ids = self.idx2id["rel"][1]
        
        print(f"# source ent : {len(s_ent_ids)}\t # target ent : {len(t_ent_ids)}")
        print(f"# source rel : {len(s_rel_ids)}\t # target rel : {len(t_rel_ids)}")
        print("------------------------------------------------------------------------------")

    

    def set_keys(self)->None:
        if "SRPRS" in self.data_path:
            translate = 2
        else:
            translate = 1
        # entity ids, idxs
        with open(self.data_path+"/ent_ids_1", encoding="utf-8") as f :
            s_ents = f.readlines()
            s_ents = [line.split("\t") for line in s_ents]
            s_ent_ids = np.array([line[0] for line in s_ents]).astype("int64")
            for s_idx, s_id in enumerate(s_ent_ids):
                self.id2idx["ent"][0][s_id] = s_idx
        self.idx2id["ent"][0] = s_ent_ids

        with open(self.data_path+"/ent_ids_2", encoding="utf-8") as f :
            t_ents = f.readlines()
            t_ents = [line.split("\t") for line in t_ents]
            t_ent_ids = np.array([line[0] for line in t_ents]).astype("int64")
            for t_idx, t_id in enumerate(t_ent_ids):
                self.id2idx["ent"][1][t_id] = t_idx
        self.idx2id["ent"][1] = t_ent_ids

        # entity names
        with open(self.data_path+"/translated_ent.txt", encoding="utf-8") as f :
            ent_names_trans = f.readlines()
            ent_names_trans = [line.strip() for line in ent_names_trans]

        ent_names = {}
        trans_ids = s_ent_ids
        no_trans = t_ents
        if translate == 2: 
            trans_ids = t_ent_ids
            no_trans = s_ents

        for e_id, ent_name in zip(trans_ids, ent_names_trans):
            ent_names[e_id] = refine_name(ent_name)
        for e_id, uri in no_trans:
            ent_names[int(e_id)] = refine_name(uri.split("/")[-1])
        self.names["ent"] = ent_names

        # relation ids, idxs
        with open(self.data_path+"/rel_ids_1", encoding="utf-8") as f :
            s_rels = f.readlines()
            s_rels = [line.split("\t") for line in s_rels]
            s_rel_ids = np.array([line[0] for line in s_rels]).astype("int64")
            for r_idx, r_id in enumerate(s_rel_ids):
                self.id2idx["rel"][0][r_id] = r_idx
        self.idx2id["rel"][0] = s_rel_ids

        with open(self.data_path+"/rel_ids_2", encoding="utf-8") as f :
            t_rels = f.readlines()
            t_rels = [line.split("\t") for line in t_rels]
            t_rel_ids = np.array([line[0] for line in t_rels]).astype("int64")    
            for r_idx, r_id in enumerate(t_rel_ids):
                self.id2idx["rel"][1][r_id] = r_idx
        self.idx2id["rel"][1] = t_rel_ids

        # relation names
        with open(self.data_path+"/translated_rel.txt", encoding="utf-8") as f :
            rel_names_trans = f.readlines()
            rel_names_trans = [line.strip() for line in rel_names_trans]

        rel_names = {}
        trans_ids = s_rel_ids
        no_trans = t_rels
        if translate == 2: 
            trans_ids = t_rel_ids
            no_trans = s_rels

        for r_id, rel_name in zip(trans_ids, rel_names_trans):
            rel_names[r_id] = refine_name(rel_name)
        for r_id, uri in no_trans:
            rel_names[int(r_id)] = refine_name(uri.split("/")[-1])
        self.names["rel"] = rel_names

        return None

    def get_feature(self, mode:str, prefix:str)->tf.Tensor:
        filename = f"{prefix}_emb.npy"
        word_vec = torch.tensor(np.load(self.data_path+filename)).float()
        word_vec = word_vec/((word_vec**2).sum(1)**0.5)[:,None]
        word_vec = word_vec.type(torch.DoubleTensor)
        feature = word_vec
        if mode=="hybrid" or prefix=="rel":
            char_vec = self.encode_bigram(prefix)
            feature = np.concatenate([word_vec, char_vec], -1)
        feature = tf.nn.l2_normalize(feature, axis=-1)
        return feature
    
    def encode_bigram(self, prefix:str):
        # generate the bigram dictionary
        if prefix=="rel" : 
            names = self.names["rel"]
        else :  
            names = self.names["ent"]
        node_size = len(names)
    
        d = {}
        count = 0
        for i in range(node_size):
            name = [names[i]]
            for word in name:
                word = word.lower()
                for idx in range(len(word)-1):
                    if word[idx:idx+2] not in d:
                        d[word[idx:idx+2]] = count
                        count += 1
        
        char_vec = np.zeros((node_size,len(d)))
        for i in range(node_size):
            name = [names[i]]
            for word in name:
                word = word.lower()
                for idx in range(len(word)-1):
                    char_vec[i,d[word[idx:idx+2]]] += 1
                
            if np.sum(char_vec[i]) == 0:
                char_vec[i] = np.random.random(len(d))-0.5
            char_vec[i] = char_vec[i]/ np.linalg.norm(char_vec[i])

        return char_vec


def refine_name(name:str)->str:
    refined = ""
    name = name.strip().replace("_"," ")
    for i in range(len(name)):
        c = name[i]
        if c.isalpha() and c.isupper() and i>0 and name[i-1].islower(): c = " "+c
        elif i>0 and name[i-1].isalpha() and c.isdecimal() : c = " "+c
        elif i>0 and name[i-1].isdecimal() and c.isalpha() : c = " "+c
        refined+=c
    return refined

def reverse_triples(triples):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i,0] = triples[i,2]
            reversed_triples[i,2] = triples[i,0]
            reversed_triples[i,1] = triples[i,1]
        return reversed_triples

def load_triples(data_path):
    with open(data_path + "triples_1") as f:
        triples1 = f.readlines()
    with open(data_path + "triples_2") as f:
        triples2 = f.readlines()
        
    triples = np.array([line.replace("\n","").split("\t") for line in triples1 + triples2]).astype(np.int64)
    node_size = max([np.max(triples[:,0]),np.max(triples[:,2])]) + 1
    rel_size = np.max(triples[:,1]) + 1
    
    all_triples = np.concatenate([triples,reverse_triples(triples)],axis=0)
    all_triples = np.unique(all_triples,axis=0)
    
    return all_triples, node_size, rel_size

def triples_to_dual_triples(triples):
        rel_dict = dict()
        for h,r,t in triples :
            if h not in rel_dict : rel_dict[h] = {r}
            else : rel_dict[h].add(r)
            if t not in rel_dict : rel_dict[t] = {r}
            else : rel_dict[t].add(r)

        dual_graph = []
        for ent_id in rel_dict.keys():
            rel_ids = list(rel_dict[ent_id])
            if len(rel_ids) > 1:
                for i in range(len(rel_ids)-1):
                    for j in range(i+1,len(rel_ids)):
                        dual_graph.append([rel_ids[i], ent_id, rel_ids[j]])
        dual_graph = np.array(dual_graph)
        dual_graph = np.concatenate([dual_graph,reverse_triples(dual_graph)],axis=0)
        dual_graph = np.unique(dual_graph,axis=0)

        node_size = max([np.max(dual_graph[:,0]),np.max(dual_graph[:,2])]) + 1
        rel_size = np.max(dual_graph[:,1]) + 1

        return dual_graph, node_size, rel_size

def load_aligned_pair(data_path):
        if "sup_pairs" not in os.listdir(data_path):
            with open(data_path + "ref_pairs") as f:
                ent_pairs = f.readlines()
        else:
            with open(data_path + "ref_pairs") as f:
                ref = f.readlines()
            with open(data_path + "sup_pairs") as f:
                sup = f.readlines()
            ent_pairs = ref + sup
            
        ent_pairs = np.array([line.replace("\n","").split("\t") for line in ent_pairs]).astype(np.int64)
        
        return ent_pairs

def get_cnt(triples):
    cnt = {}
    for h,r,t in triples:
        if h in cnt:
            cnt[h].append([r,t])
        else:
            cnt[h] = [[r,t]] 
    return cnt

def get_adjacency_matrix(triples, node_size) -> tf.SparseTensor:
    #build the relational adjacency matrix
    dr = {}
    for x,r,y in triples:
        if r not in dr:
            dr[r] = 0
        dr[r] += 1
        
    sparse_rel_matrix = []
    for i in range(node_size):
        sparse_rel_matrix.append([i,i,np.log(len(triples)/node_size)]);
    for h,r,t in triples:
        sparse_rel_matrix.append([h,t,np.log(len(triples)/dr[r])])

    sparse_rel_matrix = np.array(sorted(sparse_rel_matrix,key=lambda x:x[0]))
    sparse_rel_matrix = tf.SparseTensor(indices=sparse_rel_matrix[:,:2],values=sparse_rel_matrix[:,2],dense_shape=(node_size,node_size))
    return sparse_rel_matrix

