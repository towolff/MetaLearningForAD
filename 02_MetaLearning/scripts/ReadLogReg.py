print('Load trained LogReg..')
import joblib

model_fn = '20200303_LogRegModell.save'
filename = os.path.join(model_bib_path, model_fn)
clf = joblib.load(filename)
print(clf)