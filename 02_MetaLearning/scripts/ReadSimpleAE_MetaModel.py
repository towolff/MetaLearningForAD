print('Load Meta Model AE..')
model_name = '20200319_firstMetaModel.pt'
model_fn = os.path.join(model_bib_path, model_name)
print(model_fn)
from models.SimpleAutoEncoder import SimpleAutoEncoder
torch.manual_seed(42)

num_inpus = 17
val_lambda = 42 * 0.01

meta_model = SimpleAutoEncoder(num_inputs=num_inpus, val_lambda=val_lambda)
print(meta_model)
meta_model.load_state_dict(torch.load(model_fn))
