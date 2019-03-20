# Public python modules
import pandas as pd
import numpy as np
import pickle
import random as rnd

# default parameters
pp  = {
    'random_search_cnt_max':100,
    'learning_rate_range':[np.log10(0.0001),np.log10(100)],
    'penalty_range':[np.log10(0.0001),np.log10(100)],
    'rec_name':'random_hyperparameter.csv'
}

# Generate random hyper-parameters with size of "random_search_cnt_max"
lr = []
reg = []
for cnt in range(pp['random_search_cnt_max']):
    lr.append(10**rnd.uniform(pp['learning_rate_range'][0],pp['learning_rate_range'][1]))
    reg.append(10**rnd.uniform(pp['penalty_range'][0],pp['penalty_range'][1]))

# Save the generated hyperparameters in CSV format
record = pd.DataFrame()
record['lr'] = lr
record['reg'] = reg
record.to_csv(pp['rec_name'])