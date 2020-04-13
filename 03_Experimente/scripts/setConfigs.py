## Set Configs
print('Set configs..')
pd.options.display.max_columns = None
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

fig_path = os.path.join(os.getcwd(), 'figs')
model_path = os.path.join(os.getcwd(), 'models')
model_bib_path = os.path.join(model_path,'model_bib')
data_path = os.path.join(os.getcwd(), 'data')
exp_data_path = os.path.join(os.getcwd(), 'exp_data')
exp_models_path = os.path.join(os.getcwd(), 'exp_models')