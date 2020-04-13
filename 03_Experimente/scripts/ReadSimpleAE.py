print('+++' * 10)
print('Load untrained simple AE..')
print('--' * 5)
model_name = '20200302_firstAE_model.pt'
model_fn = os.path.join(model_bib_path, model_name)
print('Load model: {}'.format(model_fn))
print('--' * 5)

from models.SimpleAutoEncoder import SimpleAutoEncoder
torch.manual_seed(42)

num_inpus = 17
val_lambda = 42 * 0.01

model = SimpleAutoEncoder(num_inputs=num_inpus, val_lambda=val_lambda)

print(model)
print('--' * 5)

print('Init weights..')
model = model.apply(SimpleAutoEncoder.weight_init)
print('--' * 5)

print('Set model in train mode!')
model.train()


print('+++' * 10)


