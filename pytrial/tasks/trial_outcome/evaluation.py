from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score
import numpy as np
import pdb

def find_the_best_threshold(pred, label):
    best_threshold = 0
    best_f1 = 0
    for threshold in np.arange(0, 1, 0.1):
        pred_label = pred.copy()
        pred_label[pred_label >= threshold] = 1
        pred_label[pred_label < threshold] = 0
        f1score = f1_score(label, pred_label)
        if f1score > best_f1:
            best_f1 = f1score
            best_threshold = threshold
    return best_threshold

def bootstrap_evaluate(pred, label, sample_num=100):
    roc_auc_list, prauc_score_list, f1score_list = [], [], []
    for _ in range(sample_num):
        idx = np.random.choice(len(label), len(label), replace=True)
        pred_sample = pred[idx]
        label_sample = label[idx]
        res = evaluate(pred_sample, label_sample, find_best_threshold=True)
        roc_auc_list.append(res['roc_auc'])
        prauc_score_list.append(res['pr_auc'])
        f1score_list.append(res['f1'])
    print("PR-AUC   mean: "+str(np.mean(prauc_score_list))[:6], "std: "+str(np.std(prauc_score_list))[:6])
    print("F1       mean: "+str(np.mean(f1score_list))[:6], "std: "+str(np.std(f1score_list))[:6])
    print("ROC-AUC  mean: "+ str(np.mean(roc_auc_list))[:6], "std: " + str(np.std(roc_auc_list))[:6])
    return {'roc_auc': np.mean(roc_auc_list), 'pr_auc': np.mean(prauc_score_list), 'f1': np.mean(f1score_list)}

def evaluate(pred, label, find_best_threshold=False):
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(label, list):
        label = np.array(label)
    roc_auc = roc_auc_score(label, pred)
    prauc_score = average_precision_score(label, pred)
    pred_label = pred.copy()
    threshold = 0.5
    if find_best_threshold:
        threshold = find_the_best_threshold(pred, label)
    pred_label[pred_label >= threshold] = 1
    pred_label[pred_label < threshold] = 0
    f1score = f1_score(label, pred_label)
    # print('Evaluate on # of test trials: {}'.format(len(pred_label)))
    return {'pr_auc': prauc_score, 'f1': f1score, 'roc_auc': roc_auc}