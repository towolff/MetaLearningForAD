from Experiments.FineTuner import FineTuner
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from mlxtend.plotting import plot_confusion_matrix
from utils.evalUtils import calc_cm_metrics
from utils.evalUtils import print_confusion_matrix
import pandas as pd


class FineTuneExperiment:

    def __init__(self, model, fine_tune_data_x_drifted, eval_data_x_test, eval_label_x_test,
                 eval_data_x_drifted_ano, eval_label_x_drifted_ano, k, fine_tune_iterations, optimizer_name,
                 fine_tune_classes, classifier, lr, fine_tuned_model_fn, pretrained_model_fn, logreg_fn,
                 name_pretrained_model, exp_name_fn, do_debug=False):

        self.tuner = None
        self.exp_name_fn = exp_name_fn
        self.model = model
        self.fine_tune_data_x_drifted = fine_tune_data_x_drifted
        self.eval_data_x_test = eval_data_x_test
        self.eval_label_x_test = [1 if x > 0 else 0 for x in eval_label_x_test]
        self.eval_data_x_drifted_ano = eval_data_x_drifted_ano
        self.eval_label_x_drifted_ano = [1 if x > 0 else 0 for x in eval_label_x_drifted_ano]
        self.k = k
        self.lr = lr
        self.fine_tune_iterations = fine_tune_iterations
        self.optimizer = None
        self.optimizer_name = optimizer_name
        self.fine_tune_classes = fine_tune_classes
        self.classifier = classifier
        self.do_debug = do_debug
        self.fine_tuner = []
        self._init_optimizer()
        self.fine_tune_dict = {
            'optimizer': optimizer_name,
            'fine_tune_classes': str(fine_tune_classes),
            'name_pretrained_model': name_pretrained_model,
            'k': k,
            'fine_tune_iterations': fine_tune_iterations,
            'lr': lr,
            'model_fn': fine_tuned_model_fn,
            'pretrained_model_fn': pretrained_model_fn,
            'logreg_fn': logreg_fn,
            'TP_x_test': None,
            'TN_x_test': None,
            'FP_x_test': None,
            'FN_x_test': None,
            'TP_x_drifted_ano': None,
            'TN_x_drifted_ano': None,
            'FP_x_drifted_ano': None,
            'FN_x_drifted_ano': None,
            'Accuracy_x_test': None,
            'Precision_x_test': None,
            'Specifity_x_test': None,
            'Sensitivity_x_test': None,
            'Accuracy_x_drifted_ano': None,
            'Precision_x_drifted_ano': None,
            'Specifity_x_drifted_ano': None,
            'Sensitivity_x_drifted_ano': None,
        }

    def _init_optimizer(self):
        if self.optimizer_name == 'Adam' or self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'SGD' or self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

    def _start_fine_tuning(self):
        self.tuner = FineTuner(fine_tune_data=self.fine_tune_data_x_drifted, k=self.k, model=self.model,
                               optimizer=self.optimizer, fine_tune_classes=self.fine_tune_classes,
                               num_iterations=self.fine_tune_iterations)

        self.tuner.fine_tune()

    def _eval_tuned_model(self):
        # Eval on X_test
        re_tuner = []
        pred_tuner = []
        for val in self.eval_data_x_test:
            re = self.tuner.model.calc_reconstruction_error(val)
            re_tuner.append(re.item())
            re = re.reshape(1, -1)
            pred = self.classifier.predict(re.detach().numpy())
            pred_tuner.append(pred[0])

        cm = confusion_matrix(self.eval_label_x_test, pred_tuner)
        tn_tuned, fp_tuned, fn_tuned, tp_tuned = cm.ravel()
        accuracy, precision, specifity, sensitivity, _ = calc_cm_metrics(tp_tuned, tn_tuned, fp_tuned, fn_tuned)

        self.fine_tune_dict['FP_x_test'] = fp_tuned
        self.fine_tune_dict['FN_x_test'] = fn_tuned
        self.fine_tune_dict['TP_x_test'] = tp_tuned
        self.fine_tune_dict['TN_x_test'] = tn_tuned

        self.fine_tune_dict['Accuracy_x_test'] = accuracy
        self.fine_tune_dict['Precision_x_test'] = precision
        self.fine_tune_dict['Specifity_x_test'] = specifity
        self.fine_tune_dict['Sensitivity_x_test'] = sensitivity

        # Eval on X_drifted,ano
        re_tuner = []
        pred_tuner = []
        for val in self.eval_data_x_drifted_ano:
            re = self.tuner.model.calc_reconstruction_error(val)
            re_tuner.append(re.item())
            re = re.reshape(1, -1)
            pred = self.classifier.predict(re.detach().numpy())
            pred_tuner.append(pred[0])

        cm = confusion_matrix(self.eval_label_x_drifted_ano, pred_tuner)
        tn_tuned, fp_tuned, fn_tuned, tp_tuned = cm.ravel()
        accuracy, precision, specifity, sensitivity, _ = calc_cm_metrics(tp_tuned, tn_tuned, fp_tuned, fn_tuned)

        self.fine_tune_dict['FP_x_drifted_ano'] = fp_tuned
        self.fine_tune_dict['FN_x_drifted_ano'] = fn_tuned
        self.fine_tune_dict['TP_x_drifted_ano'] = tp_tuned
        self.fine_tune_dict['TN_x_drifted_ano'] = tn_tuned

        self.fine_tune_dict['Accuracy_x_drifted_ano'] = accuracy
        self.fine_tune_dict['Precision_x_drifted_ano'] = precision
        self.fine_tune_dict['Specifity_x_drifted_ano'] = specifity
        self.fine_tune_dict['Sensitivity_x_drifted_ano'] = sensitivity

    def _save_tuned_model(self):
        try:
            torch.save(self.tuner.model.state_dict(), self.fine_tune_dict['model_fn'])
            return True

        except Exception as e:
            print(e)
            return False

    def _save_experiment_data(self):
        df_exp = pd.DataFrame(self.fine_tune_dict, index=[0])
        df_exp.to_csv(self.exp_name_fn, sep=';', index=False)

    def _save(self):
        self._save_tuned_model()
        self._save_experiment_data()

    def run(self):
        self._start_fine_tuning()
        self._eval_tuned_model()
        self._save()
