'''
Create a k-partite graph
NOTE : valid edges pertain to us import only
'''

# try:
#     %load_ext autoreload
#     %autoreload 2
# except:
#     pass

from gensim.models import KeyedVectors
import os
from joblib import Parallel, delayed
import multiprocessing as mp
import pickle
from itertools import combinations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append('./../.')
sys.path.append('./../../.')
try:
    from common_utils import utils
except:
    from .common_utils import utils
ID_COL = 'PanjivRecordID'
def get_domain_dims(DATA_LOC):
    with open(os.path.join(DATA_LOC, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)
        return domain_dims


def get_edge_list(df, e_type):
    s, t = e_type.split('_')
    df_grouped = df.groupby([s, t]).size().reset_index(name='weight')
    df_grouped = df_grouped.rename(columns={s: 'source', t: 'target'})
    df_grouped['e_type'] = e_type
    return df_grouped


# ===============
# Use training data for graph creation
# DATA_LOC = './../generated_data_v1/{}'.format(DIR)
# ===============
def generate_graph_data(DIR, DATA_LOC,  train_csv_file = 'train_data.csv'):
    global ID_COL
    domain_dims = get_domain_dims(DATA_LOC)
    # ---------------------------------------
    # This data is already serialized
    # ---------------------------------------
    data_df = pd.read_csv(os.path.join(DATA_LOC, train_csv_file), index_col=None, low_memory=False)
    data_df = data_df.drop_duplicates(subset=list(domain_dims.keys()))

    # ------------------------------
    # Create graph ingestible data
    # ------------------------------
    node_types = sorted(domain_dims.keys())
    try:
        with open('./valid_edges.txt', 'r') as fh:
            valid_edge_types = fh.readlines()
            valid_edge_types = [_.strip('\n') for _ in valid_edge_types]

    except:
        valid_edge_types = ['_'.join(sorted([a, b])) for a, b in combinations(node_types, 2)]

    list_edge_df = Parallel(
        mp.cpu_count()
    )(delayed(get_edge_list)(data_df, e) for e in valid_edge_types)
    edges_df = None
    for _df in list_edge_df:
        if edges_df is not None:
            edges_df = edges_df.append(_df, ignore_index=True)
        else:
            edges_df = pd.DataFrame(_df)

    nodes_df = pd.DataFrame(columns=['ID', 'n_type'])
    idMapping_df = utils.fetch_idMappingFile(DIR)

    for domain in domain_dims.keys():
        tmp = pd.DataFrame(idMapping_df.loc[idMapping_df['domain'] == domain]['serial_id'])
        tmp = tmp.rename(columns={'serial_id': 'ID'})
        tmp['n_type'] = domain
        nodes_df = nodes_df.append(tmp, ignore_index=True)

    print('Number of nodes :: ',len(nodes_df), 'Number of edges ::', len(edges_df))

    # --------------------------
    # Save files
    # --------------------------
    SAVE_DIR = os.path.join(DATA_LOC, 'data_synthetic_graph')
    pathobj = Path(SAVE_DIR)
    pathobj.mkdir(exist_ok=True, parents=True)

    edges_df.to_csv(os.path.join(SAVE_DIR, 'edges.csv'), index=False)
    nodes_df.to_csv(os.path.join(SAVE_DIR, 'nodes.csv'), index=False)
    return

# -----------------------
# Return list of tuple (source, target, weight)
# source and target need to be type str
# -----------------------
def read_graph_data(DIR, DATA_LOC = './../generated_data_v1', train_csv_file = 'train_data.csv'):
    SOURCE_DATA_LOC = os.path.join(DATA_LOC, DIR, 'data_synthetic_graph')
    fname_e = 'edges.csv'
    fname_n = 'nodes.csv'

    # Check if file exists ::
    if not os.path.exists(os.path.join(SOURCE_DATA_LOC, fname_e)) or not os.path.exists(os.path.join(SOURCE_DATA_LOC, fname_e)):
        generate_graph_data (DIR, DATA_LOC,  train_csv_file )

    df_e = pd.read_csv(os.path.join(SOURCE_DATA_LOC, fname_e), low_memory=False, index_col=None)
    df_n = pd.read_csv(os.path.join(SOURCE_DATA_LOC, fname_n), low_memory=False, index_col=None)

    # ----------------------------------------
    # replace the node id by synthetic id
    # ----------------------------------------
    print('Types of edges', set(df_e['e_type']))

    list_tuples = []
    df_e['source'] = df_e['source'].astype(str)
    df_e['target'] = df_e['target'].astype(str)
    for i,j,k in zip(
            df_e['source'].values.tolist(), df_e['target'].values.tolist(), df_e['weight'].values.tolist()
    ):
        list_tuples.append((i,j,k))

    return list_tuples



# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
#     default='us_import1'
# )
#
# args = parser.parse_args()
# DIR = args.DIR
# main(DIR)





