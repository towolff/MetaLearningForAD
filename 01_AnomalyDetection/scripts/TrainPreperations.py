## Set Configs
print('Set configs..')
sns.set()
pd.options.display.max_columns = None
warnings.filterwarnings('ignore')
RANDOM_SEED = 42

fig_path = os.path.join(os.getcwd(), 'figs')
model_path = os.path.join(os.getcwd(), 'models')
model_bib_path = os.path.join(model_path,'model_bib')
data_path = os.path.join(os.getcwd(), 'data')

print('Read data..')
## read data
fn = "simulation_data_y_2020_2021_reduced.h5"
data_fn = os.path.join(data_path, fn)
df_data = pd.read_hdf(data_fn, key='df')
print('Shape of normal data: {}'.format(df_data.shape))

fn = 'anomalous_data_y_2022_reduced.h5'
data_fn_anormal = os.path.join(data_path, fn)
df_data_anormal = pd.read_hdf(data_fn_anormal, key='df')
print('Shape of anormal data: {}'.format(df_data_anormal.shape))
