
## Set Configs
print('Set configs..')
sns.set()
pd.options.display.max_columns = None
RANDOM_SEED = 42

fig_path = os.path.join(os.getcwd(), 'figs')
model_path = os.path.join(os.getcwd(), 'models')
model_bib_path = os.path.join(model_path,'model_bib')
data_path = os.path.join(os.getcwd(), 'data')

## read the data
print('Read the data..')
data_fn = os.path.join(data_path, 'simulation_data_y_2020_2021_reduced.h5')
df_data = pd.read_hdf(data_fn, key='df')
print('Shape of normal data (X_sim): {}'.format(df_data.shape))

data_fn_anormal = os.path.join(data_path, 'anomalous_data_y_2022_reduced.h5')
df_data_anormal = pd.read_hdf(data_fn_anormal, key='df')
print('Shape of anormal data (X_test): {}'.format(df_data_anormal.shape))

data_fn_drifted = os.path.join(data_path, 'drifted_data_y_2023_reduced_more_cos_phi.h5')
df_data_drifted = pd.read_hdf(data_fn_drifted, key='df')
print('Shape of drifted data (X_drifted): {}'.format(df_data_drifted.shape))

data_fn_drifted_anormal = os.path.join(data_path, 'anomalous_drifted_data_y_2023_reduced_more_cos_phi.h5')
df_data_drifted_anormal = pd.read_hdf(data_fn_drifted_anormal, key='df')
print('Shape of drifted anormal data (X_drifted,anormal): {}'.format(df_data_drifted_anormal.shape))

## save label
print('Save label..')
s_labels = df_data_anormal['label']
df_data_anormal.drop('label', axis=1, inplace=True)
print('Shape of anormal data (X_test): {}'.format(df_data_anormal.shape))

s_drift_labels = df_data_drifted['drift_labels']
df_data_drifted.drop('drift_labels',axis=1,inplace=True)
print('Shape of drifted data (X_drifted): {}'.format(df_data_drifted.shape))

s_drift_labels_drifted_ano = df_data_drifted_anormal['drift_labels'] 
df_data_drifted_anormal.drop('drift_labels', axis=1, inplace=True)
s_ano_labels_drifted_ano = df_data_drifted_anormal['anomaly_labels']
df_data_drifted_anormal.drop('anomaly_labels', axis=1, inplace=True)
print('Shape of drifted anormal data (X_drifted,anormal): {}'.format(df_data_drifted_anormal.shape))

### Scale data
print('Scale data..')
scaler_train = MinMaxScaler((-1,1))
scaler_train = scaler_train.fit(df_data)
scaled_anormal = scaler_train.transform(df_data_anormal.to_numpy())
scaled_normal = scaler_train.transform(df_data.to_numpy())
scaled_drifted = scaler_train.transform(df_data_drifted.to_numpy())
scaled_drifted_anormal = scaler_train.transform(df_data_drifted_anormal.to_numpy())

## prepare for PyTorch
print('Prepare data for PyTorch..')
# build tensor from numpy
anormal_torch_tensor = torch.from_numpy(scaled_anormal).type(torch.FloatTensor)
normal_torch_tensor = torch.from_numpy(scaled_normal).type(torch.FloatTensor)
drifted_torch_tensor = torch.from_numpy(scaled_drifted).type(torch.FloatTensor)
drifted_anormal_torch_tensor = torch.from_numpy(scaled_drifted_anormal).type(torch.FloatTensor)

# build TensorDataset from Tensor
anormal_dataset = TensorDataset(anormal_torch_tensor, anormal_torch_tensor)
normal_dataset = TensorDataset(normal_torch_tensor, normal_torch_tensor)
drifted_dataset = TensorDataset(drifted_torch_tensor, drifted_torch_tensor)
drifted_anormal_dataset = TensorDataset(drifted_anormal_torch_tensor, drifted_anormal_torch_tensor)

# build DataLoader from TensorDataset
anormal_dataloader = torch.utils.data.DataLoader(anormal_dataset,batch_size=128,shuffle=False, num_workers=0)
normal_dataloader = torch.utils.data.DataLoader(normal_dataset,batch_size=128,shuffle=False, num_workers=0)
drifted_dataloader = torch.utils.data.DataLoader(drifted_dataset,batch_size=128,shuffle=False, num_workers=0)
drifted_anormal_dataloader = torch.utils.data.DataLoader(drifted_anormal_dataset, batch_size=128, shuffle=False, num_workers=0)