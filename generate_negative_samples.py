import numpy as np
import os
from pandarallel import pandarallel
pandarallel.initialize()
import sys
from tqdm import tqdm

def get_domain_dims(DIR):
    return

def aux_gen(row, domain_dims, num_samples =1):
    nd = len(domain_dims)
    res = []
    for i in range(num_samples):
        row_copy = row.copy()
        num_pert = np.random.uniform(nd//2, nd)
        pert_idx = np.random.choice(list(range(nd)),num_pert,replace=False)
        for _idx in pert_idx:
            dom = domain_dims[_idx]
            row_copy[dom] = np.random.randint(0,domain_dims[dom])
        res.append(row_copy)
    return res

def generate_neg_samples(DIR):
    domain_dims = get_domain_dims(DIR)
    data_df = None
    return
