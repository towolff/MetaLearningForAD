import os
import sys
import seaborn as sns
import torch
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from Experiments.FineTuner import FineTuner
from Experiment import Experiment

print('++++' * 10)
print('+++ Preperations +++')
print('****')
print('Current CWD is: {}'.format(os.getcwd()))
os.chdir("..")
print('Switched CWD to: {}'.format(os.getcwd()))
print('****')
print('> Set configs...')
sns.set()
pd.options.display.max_columns = None
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
fig_path = os.path.join(os.getcwd(), 'figs')
model_path = os.path.join(os.getcwd(), 'models')
model_bib_path = os.path.join(model_path,'model_bib')
data_path = os.path.join(os.getcwd(), 'data')
print('****')

print('> Load data..')
data_fn = os.path.join(data_path, 'simulation_data_y_2020_2021_reduced.h5')
df_data_train = pd.read_hdf(data_fn, key='df')
print('Shape of X_train data: {}'.format(df_data_train.shape))

data_fn = os.path.join(data_path, 'anomalous_data_y_2022_reduced.h5')
df_data_anormal = pd.read_hdf(data_fn, key='df')
print('Shape of X_test data: {}'.format(df_data_anormal.shape))

s_anormal_labels = df_data_anormal['label']
df_data_anormal.drop('label', axis=1, inplace=True)
print('Shape of X_test data: {}'.format(s_anormal_labels.shape))

data_fn = os.path.join(data_path, 'drifted_data_y_2023_reduced_more_cos_phi.h5')
df_data_drifted = pd.read_hdf(data_fn, key='df')
print('Shape of X_drifted data: {}'.format(df_data_drifted.shape))

s_drift_labels_x_drifted = df_data_drifted['drift_labels']
df_data_drifted.drop('drift_labels', axis=1, inplace=True)
print('Shape of X_drifted data: {}'.format(df_data_drifted.shape))

data_fn = os.path.join(data_path, 'anomalous_drifted_data_y_2023_reduced_more_cos_phi.h5')
df_data_drifted_ano = pd.read_hdf(data_fn, key='df')
print('Shape of X_drifted,ano data: {}'.format(df_data_drifted_ano.shape))

s_drifted_ano_drift_labels = df_data_drifted_ano['drift_labels']
s_drifted_ano_ano_labels = df_data_drifted_ano['anomaly_labels']

df_data_drifted_ano.drop(['drift_labels', 'anomaly_labels'], axis=1, inplace=True)
print('Shape of X_drifted,ano data: {}'.format(df_data_drifted_ano.shape))
print('****')

print('> Scale data..')
scaler_train = MinMaxScaler((-1,1))
scaler_train = scaler_train.fit(df_data_train)
scaled_train = scaler_train.transform(df_data_train.to_numpy())
scaled_anormal = scaler_train.transform(df_data_anormal.to_numpy())
scaled_drifted = scaler_train.transform(df_data_drifted.to_numpy())
scaled_drifted_ano = scaler_train.transform(df_data_drifted_ano.to_numpy())
print('****')

print('> Build PyTorch objects..')
# build tensor from numpy
anormal_torch_tensor = torch.from_numpy(scaled_anormal).type(torch.FloatTensor)
anormal_drifted_torch_tensor = torch.from_numpy(scaled_drifted_ano).type(torch.FloatTensor)
drifted_torch_tensor_X = torch.from_numpy(scaled_drifted).type(torch.FloatTensor)

drifted_torch_tensor_y = torch.from_numpy(s_drift_labels_x_drifted.to_numpy().reshape(len(s_drift_labels_x_drifted),1)).type(torch.FloatTensor)
drifted_anormal_torch_tensor_y = torch.from_numpy(s_drifted_ano_drift_labels.to_numpy().reshape(len(s_drifted_ano_drift_labels),1)).type(torch.FloatTensor)

# build pytorch dataset from tensor
drifted_dataset = torch.utils.data.TensorDataset(drifted_torch_tensor_X, drifted_torch_tensor_y)
drifted_anormal_dataset = torch.utils.data.TensorDataset(anormal_drifted_torch_tensor, drifted_anormal_torch_tensor_y)

print('****')

print('> Load meta-trained AE..')
model_name = '20200319_firstMetaModel.pt'
model_fn = os.path.join(model_bib_path, model_name)
print(model_fn)
torch.manual_seed(42)

num_inpus = 17
val_lambda = 42 * 0.01

sys.path.append(os.getcwd())
from models.SimpleAutoEncoder import SimpleAutoEncoder

meta_model = SimpleAutoEncoder(num_inputs=num_inpus, val_lambda=val_lambda)
print(meta_model)
meta_model.load_state_dict(torch.load(model_fn))

optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
print('****')

print('> Load trained LogReg..')
import joblib

model_fn = '20200319_LogReg_MetaModel.save'
filename = os.path.join(model_bib_path, model_fn)
clf_meta = joblib.load(filename)
print(clf_meta)


print('++++' * 10)
print('+++ Start experiments +++')
NUM_ITERATIONS = 10
NUM_EXPERIMENTS = 10
K = 5
FILTERED_CLASSES = [1, 2, 3]


exp = Experiment(num_fine_tuner=NUM_EXPERIMENTS, model=meta_model, fine_tune_data=drifted_dataset,
                 eval_data=drifted_anormal_dataset, k=K, fine_tune_iterations=NUM_ITERATIONS,
                 optimizer=optimizer, fine_tune_classes=FILTERED_CLASSES, classifier=clf_meta)

exp.start_fine_tuning()

print(exp.fine_tuner)