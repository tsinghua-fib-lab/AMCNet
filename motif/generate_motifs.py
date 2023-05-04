import sys
import pdb
import pickle as pkl
import networkx as nx
import scipy
import numpy as np
import search

dataset_str='Enron'
motif1=[]
motif2=[]
motif3=[]
motif4=[]
motif5=[]
motif6=[]
motif7=[]

with open("../data/{}/{}".format(dataset_str, "adjs.pkl"), "rb") as f:
    adjs=pkl.load(f)
    for idx,adj in enumerate(adjs[:13]):
        b=np.array(adjs[idx].todense(),dtype=np.int32)
        motif1_adj= np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.int32)
        motif2_adj= np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=np.int32)
        motif3_adj= np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]],dtype=np.int32)
        motif4_adj= np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]],dtype=np.int32)
        motif5_adj= np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]],dtype=np.int32)
        motif6_adj= np.array([[0,1,1,1],[1,0,1,0],[1,1,0,1],[1,0,1,0]],dtype=np.int32)
        motif7_adj= np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]],dtype=np.int32)

        search.init_gtrie(motif1_adj, b)
        result = (search.search(b.shape[0]*b.shape[0])).reshape(b.shape[0],b.shape[0])
        motif1.append(scipy.sparse.csr_matrix(result))

        search.init_gtrie(motif2_adj, b)
        result = (search.search(b.shape[0]*b.shape[0])).reshape(b.shape[0],b.shape[0])
        motif2.append(scipy.sparse.csr_matrix(result))

        search.init_gtrie(motif3_adj, b)
        result = (search.search(b.shape[0]*b.shape[0])).reshape(b.shape[0],b.shape[0])
        motif3.append(scipy.sparse.csr_matrix(result))

        search.init_gtrie(motif4_adj, b)
        result = (search.search(b.shape[0]*b.shape[0])).reshape(b.shape[0],b.shape[0])
        motif4.append(scipy.sparse.csr_matrix(result))

        search.init_gtrie(motif5_adj, b)
        result = (search.search(b.shape[0]*b.shape[0])).reshape(b.shape[0],b.shape[0])
        motif5.append(scipy.sparse.csr_matrix(result))

        search.init_gtrie(motif6_adj, b)
        result = (search.search(b.shape[0]*b.shape[0])).reshape(b.shape[0],b.shape[0])
        motif6.append(scipy.sparse.csr_matrix(result))

        search.init_gtrie(motif7_adj, b)
        result = (search.search(b.shape[0]*b.shape[0])).reshape(b.shape[0],b.shape[0])
        motif7.append(scipy.sparse.csr_matrix(result))

with open("../data/{}/{}".format(dataset_str, "motif1.pkl"), "wb") as f:
    pkl.dump(motif1,f)
with open("../data/{}/{}".format(dataset_str, "motif2.pkl"), "wb") as f:
    pkl.dump(motif2,f)
with open("../data/{}/{}".format(dataset_str, "motif3.pkl"), "wb") as f:
    pkl.dump(motif3,f)
with open("../data/{}/{}".format(dataset_str, "motif4.pkl"), "wb") as f:
    pkl.dump(motif4,f)
with open("../data/{}/{}".format(dataset_str, "motif5.pkl"), "wb") as f:
    pkl.dump(motif5,f)
with open("../data/{}/{}".format(dataset_str, "motif6.pkl"), "wb") as f:
    pkl.dump(motif6,f)
with open("../data/{}/{}".format(dataset_str, "motif7.pkl"), "wb") as f:
    pkl.dump(motif7,f)



















