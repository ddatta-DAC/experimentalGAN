import pandas as pd
import numpy as np
import os
import sys
import argparse
import networkx as nx
from fastnode2vec import Graph, Node2Vec
import createGraph
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from operator import itemgetter
import sys
import pandas as pd
import multiprocessing as mp
from pathlib import Path
sys.path.append('./../.')
sys.path.append('./../../.')
try:
    from common_utils import utils
except:
    from .common_utils import utils


# ======================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default='us_import1'
)
parser.add_argument(
    '--EMB_DIM',
    default=64,
    type=int
)
parser.add_argument(
    '--EPOCHS',
    default=64,
    type=int
)
args = parser.parse_args()
DIR = args.DIR
epochs = args.EPOCHS
emb_dim = args.EMB_DIM

# ======================================================

data_loc = './../generated_data_v1/{}'.format(DIR)
idMapping_df = utils.fetch_idMappingFile(DIR)
EMB_SAVE_DATA_LOC = os.path.join(data_loc,'node2vec')
pathobj = Path(EMB_SAVE_DATA_LOC)
pathobj.mkdir(exist_ok=True,parents=True)

def save_vectors(w2v_model):
    global DIR
    global MODEL_SAVE_DATA_LOC
    global idMapping_df
    global emb_dim
    
    lookUp_dict = {}
    domains = set(idMapping_df['domain'])
    vectors_dict = {}
    for i,row in idMapping_df.iterrows():
        vectors_dict[row['serial_id']] = (row['entity_id'], row['domain'])

    for token, vector in w2v_model.wv.vocab.items():
        syn_id = int(token)
        e_id = lookUp_dict[syn_id][0]
        _domain = lookUp_dict[syn_id][1]
        vectors_dict[_domain][e_id] = w2v_model.wv[token]

    for n_type, _dict in vectors_dict.items():
        arr_vec = [_[1] for _ in sorted(_dict.items(), key=itemgetter(0))]
        arr_vec = np.array(arr_vec)
        fname = 'n2v_{}_{}.npy'.format(emb_dim, n_type)
        fname = os.path.join(MODEL_SAVE_DATA_LOC, fname)
        np.save(fname, arr_vec)
    return

# -----------------------------------------------
# Create a graph
graph_data = createGraph.generate_graph_data(
    DIR, DATA_LOC=data_loc
)

graph = Graph(
    graph_data,
    directed=False, weighted=True
)

n2v = Node2Vec(graph, dim=emb_dim, walk_length=100, context=10, p=1, q=1, workers=mp.cpu_count())
n2v.train(epochs=epochs)

save_vectors(n2v)