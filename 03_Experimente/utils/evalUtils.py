import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def calc_cm_metrics(tp, tn, fp, fn, debug=False):
    if debug:
        print('----' * 10)
        print('TP: {}'.format(tp))
        print('TN: {}'.format(tn))
        print('FP: {}'.format(fp))
        print('FN: {}'.format(fn))
        print('----' * 10)
    
    epsilon = 10**-8
    accuracy = ((tp + tn) / (tp + tn + fp + fn + epsilon)) * 100
    precision = (tp / (tp + fp + epsilon)) * 100
    specifity = (tn / (tn + tp + epsilon)) * 100
    sensitivity = (tp / (tp + fn + epsilon)) * 100
    f1_score = (2*tp / (2*tp + fp + fn + epsilon)) * 100
    
    return accuracy, precision, specifity, sensitivity, f1_score


def find_threshold(thresholds, model, anormal_torch_tensor, s_labels, debug=False):
    experiment = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'specifity': [],
        'sensitifity': [],
        'f1_score': [],
        'fp': [],
        'fn': [],
        'tp': [],
        'tn': []
    }

    # start experiment
    for thres in np.nditer(thresholds):
        if debug:
            print('++++' * 10)
            print('Current Threshold: {} of {}'.format(thres, max(thresholds)))

        model.set_lambda(thres)
        experiment['threshold'].append(thres)
        cnt_fp = 0
        cnt_fn = 0
        cnt_tn = 0
        cnt_tp = 0

        for vals, label in zip(anormal_torch_tensor, s_labels):
            pred = model.predict_binary(vals)
            if label == 0 and pred == 0:
                cnt_tn += 1
            elif label >= 1 and pred == 1:
                cnt_tp += 1
            elif label >= 1 and pred == 0:
                cnt_fn += 1
            elif label == 0 and pred == 1:
                cnt_fp += 1

        acc, prec, spec, sen, f1 = calc_cm_metrics(cnt_tp, cnt_tn, cnt_fp, cnt_fn)
        
        if debug:
            print('Accuracy: {}'.format(acc))
            print('Precision: {}'.format(prec))
            print('Specivity: {}'.format(spec))
            print('Sensitivity: {}'.format(sen))
        
        experiment['accuracy'].append(acc)
        experiment['precision'].append(prec)
        experiment['specifity'].append(spec)
        experiment['sensitifity'].append(sen)
        experiment['f1_score'].append(f1)
        experiment['fp'].append(cnt_fp)
        experiment['fn'].append(cnt_fn)
        experiment['tp'].append(cnt_tp)
        experiment['tn'].append(cnt_tn)
        
    
    return experiment

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7)):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d",linewidths=.6,annot_kws={"size": 25},cmap='Blues')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=28)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=28)
    plt.ylabel('Tatsächlich', fontsize=28)
    plt.xlabel('Prognostiziert',fontsize=28)
    return heatmap, fig