print('Load trained simple AE..')
model_name = '20200302_firstAE_model.pt'
model_fn = os.path.join(model_bib_path, model_name)
print(model_fn)
from models.SimpleAutoEncoder import SimpleAutoEncoder
torch.manual_seed(42)

num_inpus = 17
val_lambda = 42 * 0.01

model = SimpleAutoEncoder(num_inputs=num_inpus, val_lambda=val_lambda)
print(model)
print('Load weights..')
model.load_state_dict(torch.load(model_fn))
model.eval()

