import os
import numpy as np
import pandas as pd
import sys
import pickle
from tqdm import tqdm
from pandarallel import pandarallel
import yaml
import multiprocessing as mp


def convert_to_serializedID_format(
        target_df,
        DIR,
        data_source_loc=None,
        REFRESH=False
):
    if data_source_loc is None:
        data_source_loc = './../generated_data_v1/'
    loc = os.path.join(data_source_loc, DIR)
    with open(os.path.join(loc, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)

    idMapper_file = os.path.join(loc, 'idMapping.csv')
    if os.path.exists(idMapper_file) and REFRESH is False:
        idMapping_df = pd.read_csv(idMapper_file, index_col=None)
    else:
        cur = 0
        col_serial_id = []
        col_entity_id = []
        col_domain_names = []
        # ------------------
        for d in sorted(domain_dims.keys()):
            s = domain_dims[d]
            col_entity_id.extend(list(range(s)))
            col_domain_names.extend([d for _ in range(s)])
            tmp = np.arange(s) + cur
            tmp = tmp.tolist()
            col_serial_id.extend(tmp)
            cur += s

        data = {'domain': col_domain_names, 'entity_id': col_entity_id, 'serial_id': col_serial_id}
        idMapping_df = pd.DataFrame(data)
        # Save the idMapper
        idMapping_df.to_csv(idMapper_file, index=False)
    
    if target_df is None:
        return target_df
    # Create a dictionary for Quick access
    mapping_dict = {}
    for domain in set(idMapping_df['domain']):

        tmp =  idMapping_df.loc[(idMapping_df['domain'] == domain)]
        serial_id = tmp['serial_id'].values.tolist()
        entity_id = tmp['entity_id'].values.tolist()
        mapping_dict[domain] = {k:v for k,v in zip(entity_id,serial_id)}
    # Convert
    def convert_aux(val, domain):
        return mapping_dict[domain][val]
        # idMapping_df.loc[(idMapping_df['domain'] == domain) & (idMapping_df['entity_id'] == val)]['serial_id'].values[0]

    for domain in tqdm(list(domain_dims.keys())):
        target_df[domain] = target_df[domain].parallel_apply(convert_aux, args=(domain,))

    return target_df


def fetch_idMappingFile(DIR):
    loc = os.path.join('./../generated_data_v1',DIR)
    idMapper_file = os.path.join(loc, 'idMapping.csv')
    if os.path.exists(idMapper_file) :
        idMapping_df = pd.read_csv(idMapper_file, index_col=None)
        return idMapping_df
    else:
        return None
    

    
