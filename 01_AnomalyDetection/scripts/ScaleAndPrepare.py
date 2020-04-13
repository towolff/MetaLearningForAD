print('Save column mapping..')
column_mapping = {}
col_names = []
for idx,col in enumerate(df_data):
    if col != 'index' and col != 'label':
        col_names.append(col)
    column_mapping.update({idx:col})

print('Scale data..')
scaler_train = MinMaxScaler((-1,1))
scaler_train = scaler_train.fit(df_data)
scaled_train = scaler_train.transform(df_data.to_numpy())

print('Build PyTorch objects..')
# build tensor from numpy
train_x_torch = torch.from_numpy(scaled_train).type(torch.FloatTensor)

# build TensorDataset from Tensor
train_data = TensorDataset(train_x_torch, train_x_torch)

# build DataLoader from TensorDataset
trn_dataloader = torch.utils.data.DataLoader(train_data,batch_size=128,shuffle=False, num_workers=0)