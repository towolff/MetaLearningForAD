print('Load trained LogReg..')
import joblib

model_fn = '20200319_LogReg_MetaModel.save'
filename = os.path.join(model_bib_path, model_fn)
clf_meta = joblib.load(filename)
print(clf_meta)