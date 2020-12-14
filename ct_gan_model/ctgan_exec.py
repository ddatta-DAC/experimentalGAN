
from ctgan.synthesizer import CTGANSynthesizer
import ctgan

import numpy as np
import pandas as pd
import os
import sys
import tqdm
import pickle
import pathlib
from pathlib import Path




def get_domain_dims(DIR='us_import1'):
    with open('./generated_data_v1/{}/domain_dims.pkl'.format(DIR),'rb') as fh:
        domain_dims = pickle.load(fh)
    return domain_dims

def convert_np_to_pd(data_np, domain_dims):
    columns = list(domain_dims.keys())
    df = pd.DataFrame(data= data_np, columns=columns)
    return df, columns

real_data = np.load('./generated_data_v1/us_import1/pos_data.npy')
domain_dims = get_domain_dims()
data_df,columns = convert_np_to_pd(real_data, domain_dims)



ctgan_obj = CTGANSynthesizer()
ctgan_obj.fit(data, columns)
ctgan_obj.save('ctgan.pkl')

