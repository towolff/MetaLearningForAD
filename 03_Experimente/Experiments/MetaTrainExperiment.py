import learn2learn as l2l
import torch
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from Experiments.Learner import Learner

from sklearn.metrics import confusion_matrix


class MetaTrainExperiment:

    def __init__(self, model, train_data, eval_data, optimizer, adaption_steps, meta_lr,
                 num_iterations, k_train, num_train, num_eval, x_test, ano_label_x_test,
                 x_drifted_ano, ano_label_x_drifted_ano, model_fn, logreg_fn, exp_name):

        self.learner = None
        self.train_tasks = None
        self.eval_tasks = None
        self.trained_model = None
        self.x_test = (x_test, ano_label_x_test)
        self.x_drifted_ano = (x_drifted_ano, ano_label_x_drifted_ano)
        self.model_fn = model_fn
        self.logreg_fn = logreg_fn
        self.exp_name = exp_name
        self.exp_dict = {
            'optimizer': optimizer,
            'adaption_steps': adaption_steps,
            'meta_lr': meta_lr,
            'num_iterations': num_iterations,
            'k_train': k_train,
            'num_train': num_train,
            'num_eval': num_eval,
            'model_fn': model_fn,
            'logreg_fn': logreg_fn,
            'num_samples_x_test': len(x_test),
            'num_samples_x_ano_drifted': len(x_drifted_ano),
            'TP_x_test': None,
            'TN_x_test': None,
            'FP_x_test': None,
            'FN_x_test': None,
            'TP_x_ano_drifted': None,
            'TN_x_ano_drifted': None,
            'FP_x_ano_drifted': None,
            'FN_x_ano_drifted': None,
            'Accuracy_x_test': None,
            'Precision_x_test': None,
            'Specifity_x_test': None,
            'Sensitivity_x_test': None,
            'F1_x_test': None,
            'Accuracy_x_ano_drifted': None,
            'Precision_x_ano_drifted': None,
            'Specifity_x_ano_drifted': None,
            'Sensitivity_x_ano_drifted': None,
            'F1_x_ano_drifted': None
        }

        self._init_metadataset(train_data, eval_data, num_train, num_eval, k_train)
        self._init_learner(model, adaption_steps, meta_lr, optimizer, num_iterations)

    def _init_metadataset(self, train_data, eval_data, num_train, num_eval, k_train):
        if isinstance(train_data, l2l.data.MetaDataset) and isinstance(eval_data, l2l.data.MetaDataset):

            self.train_tasks = l2l.data.TaskDataset(train_data,
                                               task_transforms=[
                                                   l2l.data.transforms.NWays(train_data, n=1),
                                                   l2l.data.transforms.KShots(train_data, k=k_train),
                                                   l2l.data.transforms.LoadData(train_data)],
                                               num_tasks=num_train)

            self.eval_tasks = l2l.data.TaskDataset(eval_data,
                                              task_transforms=[
                                                  l2l.data.transforms.NWays(eval_data, n=1),
                                                  l2l.data.transforms.KShots(eval_data, k=1),
                                                  l2l.data.transforms.LoadData(eval_data),
                                              ],
                                              num_tasks=num_eval)
        else:
            raise Exception('train_data or eval_data is not of Type l2l.data.MetaDataset')

    def _init_learner(self, model, adaption_steps, meta_lr, optimizer, num_iterations):
        self.learner = Learner(model=model, train_task_dataset=self.train_tasks, eval_task_dataset=self.eval_tasks,
                               adaption_steps=adaption_steps, meta_lr=meta_lr, optimizer=optimizer,
                               first_order=True, num_iterations=num_iterations)

    @staticmethod
    def _calc_cm_metrics(tp, tn, fp, fn):
        epsilon = 10 ** -8
        accuracy = ((tp + tn) / (tp + tn + fp + fn + epsilon)) * 100
        precision = (tp / (tp + fp + epsilon)) * 100
        specifity = (tn / (tn + tp + epsilon)) * 100
        sensitivity = (tp / (tp + fn + epsilon)) * 100
        f1_score = (2 * tp / (2 * tp + fp + fn + epsilon)) * 100

        return accuracy, precision, specifity, sensitivity, f1_score

    @staticmethod
    def _calculate_confusion_matrix(pred, label):
        cm = confusion_matrix(label, pred)
        tn, fp, fn, tp = cm.ravel()

        return tn, fp, fn, tp

    def _train_log_reg(self):
        losses_x_test = []
        for val in self.x_test[0]:
            loss = self.trained_model.calc_reconstruction_error(val)
            losses_x_test.append(loss.item())

        s_losses_x_test = pd.Series(losses_x_test)

        x_test = s_losses_x_test.to_numpy()
        x_test = x_test.reshape(-1, 1)
        y_test = [1 if x > 0 else 0 for x in self.x_test[1]]

        self.clf = LogisticRegression(random_state=42, fit_intercept=True, solver='liblinear', class_weight={1: 2.0})
        self.clf.fit(x_test, y_test)

    def _start_training(self):
        # train AE
        self.learner.start_learning_phase()

        # Train LogReg
        if self.learner.done_training_phase:
            self.trained_model = self.learner.meta_model.module
            self._train_log_reg()

    def _predict_on_data(self, data):
        # calculate RE
        losses = []
        for val in data:
            loss = self.trained_model.calc_reconstruction_error(val)
            losses.append(loss.item())

        losses = pd.Series(losses)
        losses = losses.to_numpy()
        losses = losses.reshape(-1, 1)

        # predict
        preds = []
        for val in losses:
            val = val.reshape(1, -1)
            pred = self.clf.predict(val)
            preds.append(pred[0])

        return preds

    def _eval_meta_learned_model(self):
        if not self.learner.done_training_phase:
            raise Exception('Model not done with training!')

        predictions_x_test = self._predict_on_data(self.x_test[0])
        predictions_x_anormal_drifted = self._predict_on_data(self.x_drifted_ano[0])

        tn, fp, fn, tp = MetaTrainExperiment._calculate_confusion_matrix(predictions_x_test, self.x_test[1])
        self.exp_dict['TP_x_test'] = tp
        self.exp_dict['TN_x_test'] = tn
        self.exp_dict['FP_x_test'] = fp
        self.exp_dict['FN_x_test'] = fn

        acc, prec, spec, sen, f1 = MetaTrainExperiment._calc_cm_metrics(tp,tn,fp,fn)
        self.exp_dict['Accuracy_x_test'] = acc
        self.exp_dict['Precision_x_test'] = prec
        self.exp_dict['Specifity_x_test'] = spec
        self.exp_dict['Sensitivity_x_test'] = sen
        self.exp_dict['F1_x_test'] = f1

        tn, fp, fn, tp = MetaTrainExperiment._calculate_confusion_matrix(predictions_x_anormal_drifted, self.x_drifted_ano[1])
        self.exp_dict['TP_x_ano_drifted'] = tp
        self.exp_dict['TN_x_ano_drifted'] = tn
        self.exp_dict['FP_x_ano_drifted'] = fp
        self.exp_dict['FN_x_ano_drifted'] = fn

        acc, prec, spec, sen, f1 = MetaTrainExperiment._calc_cm_metrics(tp, tn, fp, fn)
        self.exp_dict['Accuracy_x_ano_drifted'] = acc
        self.exp_dict['Precision_x_ano_drifted'] = prec
        self.exp_dict['Specifity_x_ano_drifted'] = spec
        self.exp_dict['Sensitivity_x_ano_drifted'] = sen
        self.exp_dict['F1_x_ano_drifted'] = f1

    def _save(self):
        self._save_model()
        self._save_exp_dict()

    def _save_exp_dict(self):
        df_exp = pd.DataFrame(self.exp_dict, index=[0])
        df_exp.to_csv(self.exp_name)

    def _save_model(self):
        try:
            torch.save(self.trained_model.state_dict(), self.model_fn)
            print('Saved model at: {}'.format(self.model_fn))

            joblib.dump(self.clf, self.logreg_fn)
            print('Saved logreg at: {}'.format(self.logreg_fn))

            return True

        except Exception as e:
            print(e)
            return False

    def run(self):
        self._start_training()
        self._eval_meta_learned_model()
        self._save()
