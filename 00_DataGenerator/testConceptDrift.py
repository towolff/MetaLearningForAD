import os
import arrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp

from GridExecutor import Executor
from ConceptDrift import Drifter

print('+++ Set Configs +++')
cwd = os.getcwd()
print(cwd)
data_fn = os.path.join(cwd, 'data_modelling')
grid_fn = os.path.join(cwd, 'grid_modelling')
fig_fn = os.path.join(cwd, 'figs')
print(data_fn)
print(grid_fn)

print('+++ Load Grid +++')
fn = os.path.join(grid_fn, 'CIGRE_net.json')
grid = pp.from_json(fn)

print('+++ Load Data +++')
fn = os.path.join(data_fn, 'main_agg.h5')
all_data = pd.read_hdf(fn, key='df')
fn2 = os.path.join(data_fn, 'main.csv')
unscaled_data = pd.read_csv(fn2, sep=';', index_col='index')

print('+++ Set Intervall +++')
start_date = '2023-01-01 00:00:00'
arrw_start = arrow.get(start_date)
end_date = '2023-12-31 23:45:00'
arrw_end = arrow.get(end_date)
res = arrw_end - arrw_start
duration_days = res.days + 1
print(duration_days)

data = all_data.loc[start_date:end_date].copy()
data_unscaled = unscaled_data.loc[start_date:end_date].copy()

print('+++ Load Load Mapping +++')
fn = os.path.join(data_fn, 'load_mapping.npy')
init_load_mapping = np.load(fn, allow_pickle='TRUE').item()


drop_cols = data_unscaled.columns
data_unscaled['load_h0_normed_MW'] = data_unscaled['load_h0_normed_kW'] / 1000
data_unscaled['load_g0_normed_MW'] = data_unscaled['load_g0_normed_kW'] / 1000
data_unscaled['load_l0_normed_MW'] = data_unscaled['load_l0_normed_kW'] / 1000
data_unscaled['gen_pv_normed_MW'] = data_unscaled['gen_pv_normed_kW']
data_unscaled['gen_wind_normed_MW'] = data_unscaled['gen_wind_normed_kW']
data_unscaled['gen_gas_normed_MW'] = data_unscaled['gen_gas_normed_kW']

data_unscaled.drop(drop_cols, axis=1, inplace=True)

timestamps = [x[0] for x in data.index]
timestamps = list(dict.fromkeys(timestamps))

manipulate_switch = [
    {
        'switch_id': 4,
        'set_closed': True,
        'at_time_idx': 20
    },
    {
        'switch_id': 4,
        'set_closed': False,
        'at_time_idx': 80
    }
]

load_mapping = [
    {
        'bus_id': 1,
        'load_mapping': [3, 0, 2, 0, 1, 0],
        'at_time_idx': 323,
        'until_time_idx': 345
    },
    {
        'bus_id': 3,  # three means -> 'AGG_BUS_3'
        'load_mapping': [2, 1, 0, 0, 1, 0],
        'at_time_idx': 97,
        'until_time_idx': 5 * 96
    },
    {
        'bus_id': 10,  # three means -> 'AGG_BUS_3'
        'load_mapping': [2, 1, 1, 0, 1, 0],
        'at_time_idx': 31 * 97,
        'until_time_idx': 36 * 96
    },

]

change_cos_phi = [
    {
        'bus_id': 1,
        'load_mapping': [3, 0, 2, 0, 1, 0],  # if None, use initial load_mapping!
        'load_id_in_load_mapping': 0,
        'cos_phi': 0.95,
        'at_time_idx': 323,
        'until_time_idx': 345
    },
    {
        'bus_id': 4,
        'load_mapping': None,  # if None, use initial load_mapping!
        'load_id_in_load_mapping': 0,
        'cos_phi': 0.91,
        'at_time_idx': 400,
        'until_time_idx': 400 + 96 * 14
    }
]

concept_drift = {'manipulate_switch': manipulate_switch, 'load_mapping': load_mapping, 'change_cos_phi': change_cos_phi}

duration = duration_days * 96
print('Duration: {} 1/4 hours'.format(duration))

drifter = Drifter(grid=grid, concept_drifts=concept_drift, load_mapping=init_load_mapping, unscaled_data=data_unscaled, timestamp_list=timestamps, agg_data=data.copy())
executor = Executor(grid)

drifted_data = drifter.manipulate_cos_phi()
